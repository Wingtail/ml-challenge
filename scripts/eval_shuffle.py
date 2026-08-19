"""Required sensor-dependence check on a saved context model."""

import sys
from pathlib import Path

from har_challenge.evaluation import evaluate_saved_context


def main(seed=42):
    run = Path("outputs/reproduction/context") / f"seed_{seed}"
    evaluate_saved_context(
        run / "model.pt",
        run / "shuffle.json",
        seed=seed,
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) == 2 else 42)
