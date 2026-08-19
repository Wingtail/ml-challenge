"""Metrics and the required shuffled-embedding evaluation."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .context import ContextClassifier
from .data import load_uci_har
from .patchtst import PatchTSTEncoder


def classification_metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "per_class_f1": f1_score(labels, predictions, average=None).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            labels.extend(y.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return classification_metrics(labels, predictions)


def shuffle_sensor_embeddings(model, loader, device, repeats=10, seed=10_042):
    """Shuffle complete projected embeddings between examples without retraining."""
    model.eval()
    token_parts, label_parts = [], []
    with torch.inference_mode():
        for x, y in loader:
            token_parts.append(model.encode_sensor_token(x.to(device)).cpu())
            label_parts.append(y)
    tokens = torch.cat(token_parts)
    labels = torch.cat(label_parts)

    generator = torch.Generator().manual_seed(seed)
    runs = []
    with torch.inference_mode():
        for _ in range(repeats):
            permutation = torch.randperm(len(tokens), generator=generator)
            predictions = []
            for start in range(0, len(tokens), loader.batch_size):
                shuffled = tokens[permutation[start : start + loader.batch_size]]
                logits = model.classify_sensor_token(shuffled.to(device))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
            runs.append(classification_metrics(labels.tolist(), predictions))

    scores = [run["macro_f1"] for run in runs]
    return {
        "repeats": repeats,
        "macro_f1_mean": float(np.mean(scores)),
        "macro_f1_std": float(np.std(scores)),
        "runs": runs,
    }


def evaluate_saved_context(
    checkpoint,
    output_file,
    *,
    seed=42,
    batch_size=128,
    repeats=10,
):
    """Reload a context model and run the challenge's shuffle check."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    data = load_uci_har("dataset/UCI HAR Dataset")
    model = ContextClassifier(PatchTSTEncoder()).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(state["encoder"])
    model.projector.load_state_dict(state["projector"])
    model.head.load_state_dict(state["head"])

    loader = DataLoader(
        TensorDataset(data["x_test"], data["y_test"]),
        batch_size=batch_size,
        shuffle=False,
    )
    result = shuffle_sensor_embeddings(
        model, loader, device, repeats=repeats, seed=seed + 10_000
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2) + "\n")
    print(f"shuffled test macro-F1={100 * result['macro_f1_mean']:.2f}")
    return result
