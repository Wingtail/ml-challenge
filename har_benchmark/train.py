import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from har_benchmark import MODEL_NAMES
from har_benchmark.data import (
    AdjacentPairDataset,
    load_uci_har,
    make_dataset,
    stratified_indices,
)
from har_benchmark.models import make_pretrainer
from har_benchmark.objectives import (
    correlation,
    negative_cosine_similarity,
    nt_xent,
    random_sensor_augmentation,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=(*MODEL_NAMES, "all"), default=["all"])
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(dataset, batch_size, shuffle, args, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=drop_last,
    )


def pretrain_loader(name, data, args):
    if name == "tclhar":
        dataset = AdjacentPairDataset(data["x_train"], data["subjects_train"])
    else:
        dataset = make_dataset(data["x_train"], data["y_train"])
    return make_loader(dataset, args.batch_size, True, args, drop_last=True)


def pretrain(name, model, loader, args):
    model.to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []

    for epoch in range(args.pretrain_epochs):
        total_loss = 0.0
        sample_count = 0
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            first = batch[0].to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if name == "dctcss":
                first_view = random_sensor_augmentation(first)
                second_view = random_sensor_augmentation(first)
                prediction_first = model.online(first_view)
                prediction_second = model.online(second_view)
                with torch.no_grad():
                    target_first = model.target(first_view)
                    target_second = model.target(second_view)
                loss = 0.5 * (
                    negative_cosine_similarity(prediction_first, target_second)
                    + negative_cosine_similarity(prediction_second, target_first)
                )
            elif name == "tclhar":
                second = batch[1].to(args.device, non_blocking=True)
                loss = nt_xent(model(first), model(second), temperature=1.0)
            elif name == "autocl":
                projected, projected_augmented, augmented = model(first)
                loss = nt_xent(projected, projected_augmented, temperature=0.1)
                loss = loss + correlation(first, augmented)
            else:
                loss = model(first)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if name == "dctcss":
                model.update_target(momentum=0.99)

            total_loss += loss.item() * len(first)
            sample_count += len(first)

        mean_loss = total_loss / max(1, sample_count)
        history.append(mean_loss)
        print(f"{name} pretrain {epoch + 1:03d}/{args.pretrain_epochs}: loss={mean_loss:.5f}")
    return history


def train_linear_probe(encoder, data, args):
    encoder.to(args.device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    selected = stratified_indices(data["y_train"], args.label_fraction, args.seed)
    train_dataset = Subset(make_dataset(data["x_train"], data["y_train"]), selected.tolist())
    train_loader = make_loader(train_dataset, args.probe_batch_size, True, args)
    test_loader = make_loader(
        make_dataset(data["x_test"], data["y_test"]),
        args.probe_batch_size,
        False,
        args,
    )

    head = nn.Linear(encoder.embedding_dim, 6).to(args.device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.probe_learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.probe_epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.probe_epochs):
        head.train()
        total_loss = 0.0
        sample_count = 0
        for batch_index, (x, y) in enumerate(train_loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                embedding = encoder(x)
            logits = head(embedding)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            sample_count += len(x)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == args.probe_epochs:
            mean_loss = total_loss / max(1, sample_count)
            print(f"linear probe {epoch + 1:03d}/{args.probe_epochs}: loss={mean_loss:.5f}")

    head.eval()
    predictions = []
    labels = []
    with torch.inference_mode():
        for x, y in test_loader:
            logits = head(encoder(x.to(args.device, non_blocking=True)))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "per_class_f1": f1_score(labels, predictions, average=None).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "labeled_training_windows": len(selected),
    }
    return head, metrics


def run_model(name, data, args):
    print(f"\n=== {name} ===")
    set_seed(args.seed)
    model, encoder = make_pretrainer(name)
    loader = pretrain_loader(name, data, args)
    history = pretrain(name, model, loader, args)
    head, metrics = train_linear_probe(encoder, data, args)

    output_dir = Path(args.output_dir) / name / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "linear_probe": head.state_dict(),
            "normalization_mean": data["mean"],
            "normalization_std": data["std"],
            "embedding_dim": encoder.embedding_dim,
        },
        output_dir / "checkpoint.pt",
    )
    result = {
        "model": name,
        "seed": args.seed,
        "pretrain_epochs": args.pretrain_epochs,
        "probe_epochs": args.probe_epochs,
        "label_fraction": args.label_fraction,
        "embedding_dim": encoder.embedding_dim,
        "pretrain_loss": history,
        **metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{name}: accuracy={100 * metrics['accuracy']:.2f}% "
        f"macro_f1={100 * metrics['macro_f1']:.2f}%"
    )
    return result


def main():
    args = parse_args()
    names = list(MODEL_NAMES) if "all" in args.models else args.models
    data = load_uci_har(args.data_root)
    print(
        f"loaded UCI-HAR: train={len(data['x_train'])}, test={len(data['x_test'])}, "
        f"shape={tuple(data['x_train'].shape[1:])}, device={args.device}"
    )
    results = [run_model(name, data, args) for name in names]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
