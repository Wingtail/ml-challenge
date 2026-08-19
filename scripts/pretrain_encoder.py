"""Run the required masked PatchTST pretraining."""

import sys

from har_challenge.train import pretrain_encoder


def main(seed=42):
    pretrain_encoder(seed)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) == 2 else 42)
