"""Compare all three backends while keeping the SSL encoder fixed."""

import sys
from pathlib import Path

from har_challenge.diagnostic_training import train_diagnostic_context


def main(seed=42):
    checkpoint = Path("outputs/reproduction/pretrain") / f"seed_{seed}" / "pretrained.pt"
    output = Path("outputs/reproduction/diagnostics/frozen_encoder")
    for backend in ("pretrained", "random", "identity"):
        train_diagnostic_context(
            checkpoint,
            output / backend / f"seed_{seed}",
            seed=seed,
            backend=backend,
            freeze_encoder=True,
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) == 2 else 42)
