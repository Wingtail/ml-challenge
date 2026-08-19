"""Random-LLM and identity controls. Pass an optional seed as the first argument."""

import sys
from pathlib import Path

from har_challenge.diagnostic_training import train_diagnostic_context


def main(seed=42):
    checkpoint = Path("outputs/reproduction/pretrain") / f"seed_{seed}" / "pretrained.pt"
    output = Path("outputs/reproduction/diagnostics/backend_controls")
    for backend in ("random", "identity"):
        train_diagnostic_context(
            checkpoint,
            output / backend / f"seed_{seed}",
            seed=seed,
            backend=backend,
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) == 2 else 42)
