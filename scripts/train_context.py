"""Run the required frozen-LLM context model."""

import sys

from har_challenge.train import train_context


def main(seed=42):
    train_context(seed)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) == 2 else 42)
