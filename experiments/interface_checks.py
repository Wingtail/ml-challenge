"""MLP-projector and native-chat-template checks."""

import sys
from pathlib import Path

from har_challenge.diagnostic_training import train_diagnostic_context


def main(seed=42):
    checkpoint = Path("outputs/reproduction/pretrain") / f"seed_{seed}" / "pretrained.pt"
    output = Path("outputs/reproduction/diagnostics/interface")

    train_diagnostic_context(
        checkpoint,
        output / "mlp_projector" / f"seed_{seed}",
        seed=seed,
        projector="mlp",
    )
    for backend in ("pretrained", "random"):
        train_diagnostic_context(
            checkpoint,
            output / "chat_prompt" / backend / f"seed_{seed}",
            seed=seed,
            backend=backend,
            prompt_style="chat",
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) == 2 else 42)
