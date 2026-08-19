"""Run the required direct PatchTST classifier."""

import sys

from har_challenge.train import train_direct


def main(seed=42):
    train_direct(seed)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) == 2 else 42)
