"""The three required challenge training runs.

Edit hyperparameters in settings.py. Each public function needs only a seed.
"""

from pathlib import Path

import torch

from ._training import (
    choose_device,
    fit_classifier,
    load_encoder,
    make_loader,
    save_run,
    set_seed,
)
from .context import ContextClassifier
from .data import load_uci_har
from .evaluation import evaluate
from .patchtst import DirectClassifier, MaskedPatchPretrainer, PatchTSTEncoder
from .settings import (
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_EPOCHS,
    DATA_ROOT,
    DROPOUT,
    ENCODER_LEARNING_RATE,
    HEAD_LEARNING_RATE,
    MASK_RATIO,
    OUTPUT_ROOT,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_EPOCHS,
    PRETRAIN_LEARNING_RATE,
    WEIGHT_DECAY,
)


def run_location(stage, seed, smoke):
    root = Path("outputs/code_smoke") if smoke else OUTPUT_ROOT
    return root / stage / f"seed_{seed}"


def pretrain_encoder(seed=42, *, smoke=False):
    """Pretrain PatchTST by reconstructing masked raw-signal patches."""
    device = choose_device()
    set_seed(seed)
    data = load_uci_har(DATA_ROOT)

    epochs = 1 if smoke else PRETRAIN_EPOCHS
    batch_size = 16 if smoke else PRETRAIN_BATCH_SIZE
    max_batches = 1 if smoke else None

    encoder = PatchTSTEncoder(dropout=DROPOUT).to(device)
    model = MaskedPatchPretrainer(encoder, mask_ratio=MASK_RATIO).to(device)
    loader = make_loader(
        data["x_train"], data["y_train"], batch_size, shuffle=True, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PRETRAIN_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0

        for batch_index, (x, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * len(x)
            examples += len(x)

        history.append(total_loss / examples)
        print(f"pretrain {epoch:03d}/{epochs}: loss={history[-1]:.6f}")

    output_dir = run_location("pretrain", seed, smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "pretrained.pt"
    torch.save(
        {
            "encoder": {
                name: value.detach().cpu()
                for name, value in encoder.state_dict().items()
            },
            "mask_ratio": MASK_RATIO,
            "history": history,
            "seed": seed,
            "normalization_mean": data["mean"],
            "normalization_std": data["std"],
        },
        checkpoint,
    )
    return checkpoint


def train_direct(seed=42, *, smoke=False):
    """Fine-tune PatchTST and a linear activity-classification head."""
    device = choose_device()
    set_seed(seed)
    data = load_uci_har(DATA_ROOT)

    epochs = 1 if smoke else CLASSIFIER_EPOCHS
    batch_size = 16 if smoke else CLASSIFIER_BATCH_SIZE
    max_batches = 1 if smoke else None
    checkpoint = run_location("pretrain", seed, smoke) / "pretrained.pt"

    encoder = load_encoder(checkpoint, device, DROPOUT)
    model = DirectClassifier(encoder).to(device)
    train_loader = make_loader(
        data["x_train"], data["y_train"], batch_size, shuffle=True, device=device
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LEARNING_RATE},
            {"params": model.head.parameters(), "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    gradients = fit_classifier(
        model,
        train_loader,
        optimizer,
        epochs,
        device,
        max_batches=max_batches,
    )

    test_loader = make_loader(
        data["x_test"], data["y_test"], batch_size, shuffle=False, device=device
    )
    result = {
        "condition": "direct_classifier",
        "seed": seed,
        "test": evaluate(model, test_loader, device),
        "gradient_audit": gradients,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    save_run(run_location("direct", seed, smoke), result, model.state_dict())
    print(f"direct test macro-F1={100 * result['test']['macro_f1']:.2f}")
    return result


def train_context(seed=42, *, smoke=False):
    """Train PatchTST, projector, and head around frozen pretrained SmolLM2."""
    device = choose_device()
    set_seed(seed)
    data = load_uci_har(DATA_ROOT)

    epochs = 1 if smoke else CLASSIFIER_EPOCHS
    batch_size = 16 if smoke else CLASSIFIER_BATCH_SIZE
    max_batches = 1 if smoke else None
    checkpoint = run_location("pretrain", seed, smoke) / "pretrained.pt"

    encoder = load_encoder(checkpoint, device, DROPOUT)
    model = ContextClassifier(encoder).to(device)
    train_loader = make_loader(
        data["x_train"], data["y_train"], batch_size, shuffle=True, device=device
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LEARNING_RATE},
            {
                "params": list(model.projector.parameters())
                + list(model.head.parameters()),
                "lr": HEAD_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )
    gradients = fit_classifier(
        model,
        train_loader,
        optimizer,
        epochs,
        device,
        max_batches=max_batches,
        clip_gradients=True,
    )

    test_loader = make_loader(
        data["x_test"], data["y_test"], batch_size, shuffle=False, device=device
    )
    result = {
        "condition": "context_embedding",
        "backend": "pretrained",
        "projector": "linear",
        "prompt_style": "literal",
        "encoder_frozen": False,
        "label_fraction": 1.0,
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
    save_run(run_location("context", seed, smoke), result, state)
    print(f"context test macro-F1={100 * result['test']['macro_f1']:.2f}")
    return result
