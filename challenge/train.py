import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from challenge.data import DEFAULT_VALIDATION_SUBJECTS, load_challenge_data
from challenge.models import (
    ContinuousTokenClassifier,
    IdentityContextClassifier,
    JointEmbeddingPretrainer,
    MLPContextClassifier,
    MultiTokenTransformerClassifier,
    PatchTSTEncoder,
    PatchTSTPretrainer,
    SensorClassifier,
)
from challenge.objectives import covariance_loss, variance_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("pretrain", "probe", "finetune", "context", "all"), default="all"
    )
    parser.add_argument(
        "--method",
        choices=(
            "random",
            "masked",
            "drop_patch",
            "masked_vicreg",
            "masked_context_vcreg",
            "masked_aux",
            "masked_aux_vcreg_001",
            "masked_aux_vcreg_003",
            "masked_mlp_aux_vcreg_003",
            "vibcreg",
        ),
        default="drop_patch",
    )
    parser.add_argument("--data-root", default="dataset/UCI HAR Dataset")
    parser.add_argument("--output-dir", default="outputs/challenge")
    parser.add_argument(
        "--llm-model", default="HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    parser.add_argument(
        "--context-backend",
        choices=(
            "llm",
            "random_llm",
            "identity",
            "mlp",
            "multi_llm",
            "multi_random_llm",
            "multi_transformer",
            "channel_llm",
            "channel_random_llm",
            "channel_transformer",
            "nonlinear_llm",
        ),
        default="llm",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--context-hidden-size", type=int, default=576)
    parser.add_argument(
        "--prompt-style",
        choices=("challenge", "challenge_chat", "semantic", "scrambled", "answer"),
        default="challenge",
    )
    parser.add_argument(
        "--validation-subjects",
        type=int,
        nargs="*",
        default=DEFAULT_VALIDATION_SUBJECTS,
    )
    parser.add_argument("--final-fit", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--downstream-epochs", type=int, default=100)
    parser.add_argument("--context-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=3e-3)
    parser.add_argument("--context-learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--drop-ratio", type=float, default=0.5)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--joint-weight", type=float, default=0.01)
    parser.add_argument("--context-vcreg-weight", type=float, default=0.01)
    parser.add_argument("--sensor-aux-weight", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_encoder(dropout=0.1):
    return PatchTSTEncoder(
        channels=9,
        window_size=128,
        patch_length=8,
        stride=8,
        d_model=128,
        layers=3,
        heads=4,
        feedforward_dim=256,
        dropout=dropout,
    )


def make_loader(x, y, batch_size, shuffle, args):
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )


def selected_training_data(data, fraction: float, seed: int):
    if not 0 < fraction <= 1:
        raise ValueError("label fraction must be in (0, 1]")
    if fraction == 1:
        return data["x_train"], data["y_train"]
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in torch.unique(data["y_train"]):
        indices = torch.where(data["y_train"] == label)[0]
        count = max(1, round(len(indices) * fraction))
        selected.append(indices[torch.randperm(len(indices), generator=generator)[:count]])
    selected = torch.cat(selected)
    return data["x_train"][selected], data["y_train"][selected]


def metrics_from_predictions(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "per_class_f1": f1_score(labels, predictions, average=None).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def evaluate(model, loader, device):
    model.eval()
    labels = []
    predictions = []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())
    return metrics_from_predictions(labels, predictions)


def evaluate_sensor_head(model, loader, device):
    model.eval()
    labels = []
    predictions = []
    with torch.inference_mode():
        for x, y in loader:
            embedding = model.encoder(x.to(device, non_blocking=True))
            logits = model.sensor_head(embedding)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())
    return metrics_from_predictions(labels, predictions)


def cpu_state_dict(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def checkpoint_dir(args):
    return Path(args.output_dir) / args.method / f"seed_{args.seed}"


def load_initial_encoder(args):
    encoder = make_encoder(args.dropout)
    if args.method != "random":
        path = (
            Path(args.pretrained_checkpoint)
            if args.pretrained_checkpoint
            else checkpoint_dir(args) / "pretrained.pt"
        )
        if not path.exists():
            raise FileNotFoundError(f"run --stage pretrain first; missing {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])
    return encoder


def pretrain(data, args):
    if args.method == "random":
        print("random initialization has no self-supervised pretraining stage")
        return None

    encoder = make_encoder(args.dropout)
    if args.method in {
        "masked_context_vcreg",
        "masked_aux",
        "masked_aux_vcreg_001",
        "masked_aux_vcreg_003",
        "masked_mlp_aux_vcreg_003",
    }:
        print(f"{args.method} reuses a masked pretrained checkpoint")
        return None
    drop_ratio = args.drop_ratio if args.method == "drop_patch" else 0.0
    if args.method == "masked_vicreg":
        model = JointEmbeddingPretrainer(
            encoder,
            covariance="vicreg",
            reconstruction_weight=1.0,
            representation_weight=args.joint_weight,
            mask_ratio=args.mask_ratio,
        ).to(args.device)
    elif args.method == "vibcreg":
        model = JointEmbeddingPretrainer(
            encoder,
            covariance="vibcreg",
            reconstruction_weight=0.0,
            representation_weight=1.0,
            mask_ratio=args.mask_ratio,
        ).to(args.device)
    else:
        model = PatchTSTPretrainer(encoder, drop_ratio, args.mask_ratio).to(args.device)
    loader = make_loader(
        data["x_train"], data["y_train"], args.batch_size, True, args
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    for epoch in range(args.pretrain_epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch_index, (x, _) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            x = x.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, JointEmbeddingPretrainer):
                loss, parts = model(x, return_details=True)
            else:
                loss = model(x)
                parts = None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
            sample_count += len(x)
        mean_loss = total_loss / max(sample_count, 1)
        history.append(mean_loss)
        message = (
            f"{args.method} pretrain {epoch + 1:03d}/{args.pretrain_epochs}: "
            f"loss={mean_loss:.6f}"
        )
        if parts is not None:
            message += (
                f" recon={parts['reconstruction'].item():.4f}"
                f" inv={parts['invariance'].item():.4f}"
                f" var={parts['variance'].item():.4f}"
                f" cov={parts['covariance'].item():.4f}"
            )
        print(message)

    output = checkpoint_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": cpu_state_dict(encoder),
            "pretraining_model": cpu_state_dict(model),
            "drop_ratio": drop_ratio,
            "mask_ratio": args.mask_ratio,
            "joint_weight": args.joint_weight,
            "pretrain_loss": history,
            "normalization_mean": data["mean"],
            "normalization_std": data["std"],
            "validation_subjects": tuple(args.validation_subjects),
            "pretrain_epochs": args.pretrain_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seed": args.seed,
        },
        output / "pretrained.pt",
    )
    return history


def train_sensor_classifier(data, args, frozen: bool):
    encoder = load_initial_encoder(args)
    model = SensorClassifier(encoder).to(args.device)
    if frozen:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    x_train, y_train = selected_training_data(data, args.label_fraction, args.seed)
    train_loader = make_loader(x_train, y_train, args.batch_size, True, args)
    validation_loader = None
    if not args.final_fit:
        validation_loader = make_loader(
            data["x_validation"], data["y_validation"], args.batch_size, False, args
        )
    test_loader = make_loader(data["x_test"], data["y_test"], args.batch_size, False, args)
    learning_rate = args.probe_learning_rate if frozen else args.finetune_learning_rate
    encoder_learning_rate = (
        args.encoder_learning_rate
        if args.encoder_learning_rate is not None
        else learning_rate
    )
    if frozen:
        optimizer_groups = [{"params": model.head.parameters(), "lr": learning_rate}]
    else:
        optimizer_groups = [
            {"params": model.encoder.parameters(), "lr": encoder_learning_rate},
            {"params": model.head.parameters(), "lr": learning_rate},
        ]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_state = None
    stale_epochs = 0
    best_epoch = None
    gradient_audit = None

    for epoch in range(args.downstream_epochs):
        model.train()
        if frozen:
            model.encoder.eval()
        total_loss = 0.0
        sample_count = 0
        for batch_index, (x, y) in enumerate(train_loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if gradient_audit is None:
                gradient_audit = {
                    "encoder": module_gradient_norm(model.encoder),
                    "head": module_gradient_norm(model.head),
                }
            optimizer.step()
            total_loss += loss.item() * len(x)
            sample_count += len(x)

        mean_loss = total_loss / max(sample_count, 1)
        message = (
            f"{'probe' if frozen else 'finetune'} {epoch + 1:03d}/"
            f"{args.downstream_epochs}: loss={mean_loss:.5f}"
        )
        if args.final_fit:
            best_state = cpu_state_dict(model)
            best_epoch = epoch + 1
        else:
            validation = evaluate(model, validation_loader, args.device)
            message += f" val_f1={validation['macro_f1']:.4f}"
            if validation["macro_f1"] > best_f1:
                best_f1 = validation["macro_f1"]
                best_state = cpu_state_dict(model)
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
        print(message)
        if not args.final_fit and stale_epochs >= args.patience:
            print(f"early stopping after {epoch + 1} epochs")
            break

    model.load_state_dict(best_state)
    name = "linear_probe" if frozen else "finetune"
    result = {
        "stage": f"final_{name}" if args.final_fit else name,
        "method": args.method,
        "seed": args.seed,
        "label_fraction": args.label_fraction,
        "labeled_training_windows": len(x_train),
        "maximum_epochs": args.downstream_epochs,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
        "learning_rate": learning_rate,
        "encoder_learning_rate": encoder_learning_rate,
        "head_learning_rate": learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "training_subjects": sorted(torch.unique(data["subjects_train"]).tolist()),
        "validation_subjects": sorted(
            torch.unique(data["subjects_validation"]).tolist()
        ),
        "trainable_parameters": {
            "encoder": sum(
                parameter.numel()
                for parameter in model.encoder.parameters()
                if parameter.requires_grad
            ),
            "head": sum(parameter.numel() for parameter in model.head.parameters()),
        },
        "gradient_audit": gradient_audit,
    }
    if not args.final_fit:
        result["validation"] = evaluate(model, validation_loader, args.device)
    if not args.validation_only:
        result["test"] = evaluate(model, test_loader, args.device)
    output = checkpoint_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / f"{name}.pt")
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.validation_only:
        print(f"{args.method} {name}: validation macro_f1={100 * validation['macro_f1']:.2f}%")
    else:
        test = result["test"]
        print(
            f"{args.method} {name}: test accuracy={100 * test['accuracy']:.2f}% "
            f"macro_f1={100 * test['macro_f1']:.2f}%"
        )
    return result


def context_state_dict(model):
    state = {
        "encoder": cpu_state_dict(model.encoder),
        "projector": cpu_state_dict(model.projector),
        "head": cpu_state_dict(model.head),
        "sensor_head": cpu_state_dict(model.sensor_head),
    }
    if hasattr(model, "context_model"):
        state["context_model"] = cpu_state_dict(model.context_model)
        state["context_norm"] = cpu_state_dict(model.norm)
        state["context_query"] = model.query.detach().cpu().clone()
    return state


def load_context_state_dict(model, state):
    model.encoder.load_state_dict(state["encoder"])
    model.projector.load_state_dict(state["projector"])
    model.head.load_state_dict(state["head"])
    if "sensor_head" in state:
        model.sensor_head.load_state_dict(state["sensor_head"])
    if "context_model" in state:
        model.context_model.load_state_dict(state["context_model"])
        model.norm.load_state_dict(state["context_norm"])
        model.query.data.copy_(state["context_query"].to(model.query.device))


def context_regularization_weight(args):
    if args.method == "masked_aux_vcreg_001":
        return 0.001
    if args.method in {"masked_aux_vcreg_003", "masked_mlp_aux_vcreg_003"}:
        return 0.003
    if args.method == "masked_context_vcreg":
        return args.context_vcreg_weight
    return 0.0


def context_optimizer_groups(model, encoder_learning_rate, adapter_learning_rate):
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    adapter_parameters = []
    for module in (model.projector, model.head, model.sensor_head):
        adapter_parameters.extend(
            parameter for parameter in module.parameters() if parameter.requires_grad
        )
    if hasattr(model, "context_parameters"):
        adapter_parameters.extend(model.context_parameters())
    return [
        {
            "name": "encoder",
            "params": encoder_parameters,
            "lr": encoder_learning_rate,
        },
        {
            "name": "projector_and_heads",
            "params": adapter_parameters,
            "lr": adapter_learning_rate,
        },
    ]


def module_gradient_norm(module):
    squared_norm = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().square().sum().item()
    return squared_norm**0.5


def shuffled_context_metrics(model, loader, device, repeats, seed):
    model.eval()
    tokens = []
    labels = []
    with torch.inference_mode():
        for x, y in loader:
            tokens.append(model.encode_sensor_token(x.to(device, non_blocking=True)).cpu())
            labels.append(y)
    tokens = torch.cat(tokens)
    labels = torch.cat(labels)

    results = []
    generator = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        for _ in range(repeats):
            permutation = torch.randperm(len(tokens), generator=generator)
            predictions = []
            for start in range(0, len(tokens), loader.batch_size):
                shuffled = tokens[permutation[start : start + loader.batch_size]].to(device)
                predictions.extend(
                    model.classify_sensor_token(shuffled).argmax(dim=1).cpu().tolist()
                )
            results.append(metrics_from_predictions(labels.tolist(), predictions))
    return {
        "repeats": repeats,
        "accuracy_mean": float(np.mean([item["accuracy"] for item in results])),
        "accuracy_std": float(np.std([item["accuracy"] for item in results], ddof=0)),
        "macro_f1_mean": float(np.mean([item["macro_f1"] for item in results])),
        "macro_f1_std": float(np.std([item["macro_f1"] for item in results], ddof=0)),
        "runs": results,
    }


def train_context_model(data, args):
    if not args.device.startswith("cuda"):
        raise ValueError("the SmolLM2 experiment is intended to run on CUDA")
    encoder = load_initial_encoder(args)
    if args.context_backend == "mlp" or args.method == "masked_mlp_aux_vcreg_003":
        model = MLPContextClassifier(
            encoder, hidden_size=args.context_hidden_size
        ).to(args.device)
        readout_name = "mlp"
    elif args.context_backend == "identity":
        model = IdentityContextClassifier(
            encoder, hidden_size=args.context_hidden_size
        ).to(args.device)
        readout_name = "identity"
    elif args.context_backend in {"multi_transformer", "channel_transformer"}:
        model = MultiTokenTransformerClassifier(
            encoder,
            dropout=args.dropout,
            token_layout=(
                "channel"
                if args.context_backend == "channel_transformer"
                else "temporal"
            ),
        ).to(args.device)
        readout_name = f"two_layer_{model.token_layout}_sensor_transformer"
    else:
        model = ContinuousTokenClassifier.from_pretrained(
            encoder,
            model_name=args.llm_model,
            dtype=torch.bfloat16,
            prompt_style=args.prompt_style,
            random_weights=args.context_backend
            in {"random_llm", "multi_random_llm", "channel_random_llm"},
            temporal_tokens=args.context_backend in {"multi_llm", "multi_random_llm"},
            channel_tokens=args.context_backend
            in {"channel_llm", "channel_random_llm"},
            nonlinear_projector=args.context_backend == "nonlinear_llm",
        ).to(args.device)
        readout_name = (
            f"random:{args.llm_model}"
            if "random_llm" in args.context_backend
            else args.llm_model
        )
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    use_auxiliary = "_aux" in args.method
    if not use_auxiliary:
        for parameter in model.sensor_head.parameters():
            parameter.requires_grad = False
    x_train, y_train = selected_training_data(data, args.label_fraction, args.seed)
    train_loader = make_loader(
        x_train, y_train, args.context_batch_size, True, args
    )
    validation_loader = None
    if not args.final_fit:
        validation_loader = make_loader(
            data["x_validation"],
            data["y_validation"],
            args.context_batch_size,
            False,
            args,
        )
    test_loader = make_loader(
        data["x_test"], data["y_test"], args.context_batch_size, False, args
    )
    if hasattr(model, "language_model") and any(
        parameter.requires_grad for parameter in model.language_model.parameters()
    ):
        raise RuntimeError("the language model must remain frozen")
    encoder_learning_rate = (
        args.encoder_learning_rate
        if args.encoder_learning_rate is not None
        else args.context_learning_rate
    )
    optimizer_groups = context_optimizer_groups(
        model, encoder_learning_rate, args.context_learning_rate
    )
    trainable = [
        parameter for group in optimizer_groups for parameter in group["params"]
    ]
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_state = None
    stale_epochs = 0
    best_epoch = None
    gradient_audit = None
    regularization_weight = context_regularization_weight(args)

    for epoch in range(args.context_epochs):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        total_loss = 0.0
        sample_count = 0
        for batch_index, (x, y) in enumerate(train_loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "encode_context_input"):
                sensor_embedding, token = model.encode_context_input(x)
            else:
                sensor_embedding = model.encoder(x)
                token = model.project_sensor_embedding(sensor_embedding)
            if gradient_audit is None:
                token.retain_grad()
            hidden = model.contextualize_sensor_token(token)
            logits = model.head(hidden)
            task_loss = criterion(logits, y)
            auxiliary_loss = (
                criterion(model.sensor_head(sensor_embedding), y)
                if use_auxiliary
                else None
            )
            regularization = (
                25.0 * variance_loss(hidden) + covariance_loss(hidden)
                if regularization_weight > 0
                else None
            )
            loss = task_loss
            if auxiliary_loss is not None:
                loss = loss + args.sensor_aux_weight * auxiliary_loss
            if regularization is not None:
                loss = loss + regularization_weight * regularization
            loss.backward()
            if gradient_audit is None:
                language_model_gradient_norm = (
                    module_gradient_norm(model.language_model)
                    if hasattr(model, "language_model")
                    else None
                )
                gradient_audit = {
                    "sensor_token": (
                        float(token.grad.detach().float().norm().item())
                        if token.grad is not None
                        else None
                    ),
                    "encoder": module_gradient_norm(model.encoder),
                    "projector": module_gradient_norm(model.projector),
                    "head": module_gradient_norm(model.head),
                    "sensor_head": module_gradient_norm(model.sensor_head),
                    "language_model": language_model_gradient_norm,
                }
                required = ("sensor_token", "projector", "head")
                if not args.freeze_encoder:
                    required += ("encoder",)
                if any(not gradient_audit[name] for name in required):
                    raise RuntimeError(f"missing required gradient: {gradient_audit}")
                if language_model_gradient_norm not in (None, 0.0):
                    raise RuntimeError("frozen language-model parameters received gradients")
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
            sample_count += len(x)

        mean_loss = total_loss / max(sample_count, 1)
        message = (
            f"context {epoch + 1:03d}/{args.context_epochs}: "
            f"loss={mean_loss:.5f}"
            + (
                f" task={task_loss.item():.4f}"
                f" aux={auxiliary_loss.item():.4f}"
                if auxiliary_loss is not None
                else ""
            )
            + (
                f" vcreg={regularization.item():.4f}"
                if regularization is not None
                else ""
            )
        )
        if args.final_fit:
            best_state = copy.deepcopy(context_state_dict(model))
            best_epoch = epoch + 1
        else:
            validation = evaluate(model, validation_loader, args.device)
            message += f" val_f1={validation['macro_f1']:.4f}"
            if validation["macro_f1"] > best_f1:
                best_f1 = validation["macro_f1"]
                best_state = copy.deepcopy(context_state_dict(model))
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
        print(message)
        if not args.final_fit and stale_epochs >= args.patience:
            print(f"early stopping after {epoch + 1} epochs")
            break

    load_context_state_dict(model, best_state)
    result = {
        "stage": "final_context" if args.final_fit else "context",
        "method": args.method,
        "seed": args.seed,
        "llm_model": readout_name,
        "context_backend": args.context_backend,
        "encoder_frozen": args.freeze_encoder,
        "prompt_style": args.prompt_style,
        "label_fraction": args.label_fraction,
        "labeled_training_windows": len(x_train),
        "training_subjects": sorted(torch.unique(data["subjects_train"]).tolist()),
        "validation_subjects": sorted(
            torch.unique(data["subjects_validation"]).tolist()
        ),
        "maximum_epochs": args.context_epochs,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
        "batch_size": args.context_batch_size,
        "learning_rate": args.context_learning_rate,
        "encoder_learning_rate": encoder_learning_rate,
        "adapter_learning_rate": args.context_learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "context_vcreg_weight": regularization_weight,
        "sensor_aux_weight": args.sensor_aux_weight if use_auxiliary else 0.0,
        "trainable_parameters": {
            "encoder": sum(parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad),
            "projector": sum(parameter.numel() for parameter in model.projector.parameters() if parameter.requires_grad),
            "head": sum(parameter.numel() for parameter in model.head.parameters() if parameter.requires_grad),
            "sensor_head": sum(parameter.numel() for parameter in model.sensor_head.parameters() if parameter.requires_grad),
            "language_model": sum(
                parameter.numel()
                for parameter in getattr(model, "language_model", nn.Module()).parameters()
                if parameter.requires_grad
            ),
            "context_model": sum(
                parameter.numel()
                for parameter in model.context_parameters()
            )
            if hasattr(model, "context_parameters")
            else 0,
        },
        "gradient_audit": gradient_audit,
    }
    if not args.final_fit:
        result["validation"] = evaluate(model, validation_loader, args.device)
        if use_auxiliary:
            result["validation_sensor_head"] = evaluate_sensor_head(
                model, validation_loader, args.device
            )
    if not args.validation_only:
        result["test"] = evaluate(model, test_loader, args.device)
        if use_auxiliary:
            result["test_sensor_head"] = evaluate_sensor_head(
                model, test_loader, args.device
            )
        result["shuffled_test"] = shuffled_context_metrics(
            model, test_loader, args.device, args.shuffle_repeats, args.seed + 10_000
        )
    output = checkpoint_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    best_state["normalization_mean"] = data["mean"]
    best_state["normalization_std"] = data["std"]
    torch.save(best_state, output / "context.pt")
    (output / "context.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.validation_only:
        validation = result["validation"]
        print(f"{args.method} context: validation macro_f1={100 * validation['macro_f1']:.2f}%")
    else:
        test = result["test"]
        shuffled = result["shuffled_test"]
        print(
            f"{args.method} context: test accuracy={100 * test['accuracy']:.2f}% "
            f"macro_f1={100 * test['macro_f1']:.2f}%; shuffled macro_f1="
            f"{100 * shuffled['macro_f1_mean']:.2f}%"
        )
    return result


def main():
    args = parse_args()
    if args.final_fit and args.validation_only:
        raise ValueError("--final-fit and --validation-only cannot be combined")
    if args.final_fit and args.stage not in {
        "pretrain",
        "probe",
        "finetune",
        "context",
    }:
        raise ValueError(
            "--final-fit supports pretrain, probe, finetune, or context stages"
        )
    set_seed(args.seed)
    validation_subjects = () if args.final_fit else tuple(args.validation_subjects)
    args.validation_subjects = validation_subjects
    data = load_challenge_data(args.data_root, validation_subjects=validation_subjects)
    print(
        f"raw UCI-HAR signals: train={len(data['x_train'])}, "
        f"validation={len(data['x_validation'])}, test={len(data['x_test'])}, "
        f"shape={tuple(data['x_train'].shape[1:])}, device={args.device}"
    )
    print(
        f"validation subjects={sorted(torch.unique(data['subjects_validation']).tolist())}; "
        f"official test subjects={sorted(torch.unique(data['subjects_test']).tolist())}"
    )

    if args.stage in ("pretrain", "all"):
        pretrain(data, args)
    if args.stage in ("probe", "all"):
        train_sensor_classifier(data, args, frozen=True)
    if args.stage in ("finetune", "all"):
        train_sensor_classifier(data, args, frozen=False)
    if args.stage in ("context", "all"):
        train_context_model(data, args)


if __name__ == "__main__":
    main()
