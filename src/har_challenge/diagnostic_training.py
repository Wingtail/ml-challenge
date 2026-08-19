"""Flexible training used only by the secondary diagnostic experiments."""

from pathlib import Path

import torch

from ._training import (
    choose_device,
    fit_classifier,
    load_encoder,
    make_loader,
    save_run,
    select_label_fraction,
    set_seed,
)
from .data import load_uci_har
from .diagnostic_context import build_diagnostic_context
from .evaluation import evaluate


def train_diagnostic_context(
    pretrained_checkpoint,
    output_dir,
    *,
    seed=42,
    backend="pretrained",
    projector="linear",
    prompt_style="literal",
    freeze_encoder=False,
    label_fraction=1.0,
):
    """Run one controlled variation of the required context model."""
    device = choose_device()
    set_seed(seed)
    data = load_uci_har("dataset/UCI HAR Dataset")
    encoder = load_encoder(pretrained_checkpoint, device, dropout=0.1)
    model = build_diagnostic_context(
        encoder,
        backend=backend,
        projector=projector,
        prompt_style=prompt_style,
    ).to(device)

    if freeze_encoder:
        model.encoder.requires_grad_(False)

    train_x, train_y = select_label_fraction(data, label_fraction, seed)
    train_loader = make_loader(
        train_x, train_y, batch_size=128, shuffle=True, device=device
    )

    projector_and_head = list(model.projector.parameters()) + list(
        model.head.parameters()
    )
    parameter_groups = [{"params": projector_and_head, "lr": 1e-3}]
    if not freeze_encoder:
        parameter_groups.insert(
            0, {"params": model.encoder.parameters(), "lr": 1e-4}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
    gradients = fit_classifier(
        model,
        train_loader,
        optimizer,
        epochs=18,
        device=device,
        clip_gradients=True,
    )

    test_loader = make_loader(
        data["x_test"], data["y_test"], batch_size=128, shuffle=False, device=device
    )
    result = {
        "condition": "context_embedding",
        "backend": backend,
        "projector": projector,
        "prompt_style": prompt_style,
        "encoder_frozen": freeze_encoder,
        "label_fraction": label_fraction,
        "seed": seed,
        "test": evaluate(model, test_loader, device),
        "gradient_audit": gradients,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
    state = {
        "encoder": model.encoder.state_dict(),
        "projector": model.projector.state_dict(),
        "head": model.head.state_dict(),
    }
    save_run(Path(output_dir), result, state)
    print(f"context test macro-F1={100 * result['test']['macro_f1']:.2f}")
    return result
