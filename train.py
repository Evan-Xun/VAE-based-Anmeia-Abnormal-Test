"""Compatibility entry point for the current CBC VAE training script."""

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from src.train import parse_args, train


if __name__ == "__main__":
    train(parse_args())
