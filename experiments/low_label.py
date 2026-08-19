"""Frozen-encoder low-label matrix used in the LLM-value study."""

import sys
from pathlib import Path

from har_challenge.diagnostic_training import train_diagnostic_context


LABEL_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
BACKENDS = ("pretrained", "random", "identity")


def main(seed=42):
    checkpoint = Path("outputs/reproduction/pretrain") / f"seed_{seed}" / "pretrained.pt"
    output = Path("outputs/reproduction/diagnostics/low_label")
    for fraction in LABEL_FRACTIONS:
        for backend in BACKENDS:
            train_diagnostic_context(
                checkpoint,
                output / str(fraction) / backend / f"seed_{seed}",
                seed=seed,
                backend=backend,
                freeze_encoder=True,
                label_fraction=fraction,
            )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) == 2 else 42)
