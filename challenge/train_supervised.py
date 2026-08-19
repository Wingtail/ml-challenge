"""Train simple classifiers directly on the raw UCI HAR signals."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from challenge.data import DEFAULT_VALIDATION_SUBJECTS, load_challenge_data
from challenge.models import FullyConvolutionalClassifier, SensorClassifier
from challenge.train import evaluate, make_encoder, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("fcn", "patchtst"), required=True)
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--output-dir", default="outputs/supervised")
    parser.add_argument(
        "--validation-subjects",
        type=int,
        nargs="*",
        default=DEFAULT_VALIDATION_SUBJECTS,
    )
    parser.add_argument("--final-fit", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_model(architecture: str, dropout: float):
    if architecture == "fcn":
        return FullyConvolutionalClassifier()
    return SensorClassifier(make_encoder(dropout))


def make_loader(x, y, args, shuffle):
    return DataLoader(
        TensorDataset(x, y),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )


def train(args):
    set_seed(args.seed)
    data = load_challenge_data(args.data_root, tuple(args.validation_subjects))
    model = make_model(args.architecture, args.dropout).to(args.device)
    train_loader = make_loader(data["x_train"], data["y_train"], args, True)
    validation_loader = None
    if not args.final_fit:
        validation_loader = make_loader(
            data["x_validation"], data["y_validation"], args, False
        )
    test_loader = make_loader(data["x_test"], data["y_test"], args, False)

    # AdamW with zero weight decay is Adam. Keeping the same optimizer class as
    # the existing experiment avoids an irrelevant implementation difference.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)

        mean_loss = total_loss / len(data["x_train"])
        message = f"epoch {epoch:03d}/{args.epochs}: loss={mean_loss:.5f}"
        if args.final_fit:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            validation = evaluate(model, validation_loader, args.device)
            message += f" val_f1={validation['macro_f1']:.4f}"
            if validation["macro_f1"] > best_f1:
                best_f1 = validation["macro_f1"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
        print(message)
        if not args.final_fit and stale_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    result = {
        "architecture": args.architecture,
        "initialization": "random",
        "self_supervised_pretraining": False,
        "seed": args.seed,
        "training_subjects": sorted(torch.unique(data["subjects_train"]).tolist()),
        "validation_subjects": sorted(
            torch.unique(data["subjects_validation"]).tolist()
        ),
        "training_windows": len(data["x_train"]),
        "maximum_epochs": args.epochs,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "dropout": args.dropout if args.architecture == "patchtst" else 0.0,
        "weight_decay": 0.0,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    if not args.final_fit:
        result["validation"] = evaluate(model, validation_loader, args.device)
    if not args.validation_only:
        result["test"] = evaluate(model, test_loader, args.device)

    run_name = (
        f"seed_{args.seed}_lr_{args.learning_rate:g}_batch_{args.batch_size}"
    )
    output = Path(args.output_dir) / args.architecture / run_name
    output.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in best_state.items()}
    torch.save(state, output / "model.pt")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    metric_name = "validation" if args.validation_only else "test"
    print(
        f"{args.architecture} {metric_name} macro_f1="
        f"{100 * result[metric_name]['macro_f1']:.2f}% at epoch {best_epoch}"
    )
    return result


if __name__ == "__main__":
    train(parse_args())
