"""Editable settings for the three required challenge runs."""

from pathlib import Path


DATA_ROOT = Path("dataset/UCI HAR Dataset")
OUTPUT_ROOT = Path("outputs/reproduction")

PRETRAIN_EPOCHS = 100
PRETRAIN_BATCH_SIZE = 256
PRETRAIN_LEARNING_RATE = 1e-3
MASK_RATIO = 0.4

CLASSIFIER_EPOCHS = 18
CLASSIFIER_BATCH_SIZE = 128
ENCODER_LEARNING_RATE = 1e-4
HEAD_LEARNING_RATE = 1e-3

DROPOUT = 0.1
WEIGHT_DECAY = 0.0
