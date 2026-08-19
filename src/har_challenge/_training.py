"""Small training helpers shared by required and diagnostic runs."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .patchtst import PatchTSTEncoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(x, y, batch_size, shuffle, device):
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device == "cuda",
    )


def load_encoder(checkpoint, device, dropout):
    encoder = PatchTSTEncoder(dropout=dropout)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["encoder"])
    return encoder.to(device)


def save_run(output_dir, result, state):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / "model.pt")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def gradient_norm(module):
    if module is None:
        return None
    squared_norm = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().square().sum().item()
    return squared_norm**0.5


def fit_classifier(
    model,
    loader,
    optimizer,
    epochs,
    device,
    *,
    max_batches=None,
    clip_gradients=False,
):
    loss_function = nn.CrossEntropyLoss()
    first_batch_gradients = None

    for epoch in range(1, epochs + 1):
        model.train()
        if not any(parameter.requires_grad for parameter in model.encoder.parameters()):
            model.encoder.eval()

        total_loss = 0.0
        examples = 0
        for batch_index, (x, y) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x), y)
            loss.backward()

            if first_batch_gradients is None:
                first_batch_gradients = {
                    "encoder": gradient_norm(model.encoder),
                    "projector": gradient_norm(getattr(model, "projector", None)),
                    "head": gradient_norm(model.head),
                    "language_model": gradient_norm(
                        getattr(model, "language_model", None)
                    ),
                }

            if clip_gradients:
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()

            total_loss += loss.item() * len(x)
            examples += len(x)

        print(f"train {epoch:03d}/{epochs}: loss={total_loss / examples:.6f}")

    return first_batch_gradients


def select_label_fraction(data, fraction, seed):
    if fraction == 1.0:
        return data["x_train"], data["y_train"]

    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in torch.unique(data["y_train"]):
        indices = torch.where(data["y_train"] == label)[0]
        count = max(1, round(len(indices) * fraction))
        order = torch.randperm(len(indices), generator=generator)
        selected.append(indices[order[:count]])
    selected = torch.cat(selected)
    return data["x_train"][selected], data["y_train"][selected]
