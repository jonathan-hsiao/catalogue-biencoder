#!/usr/bin/env python3
"""
CLI entry point for training.
"""
import sys
from pathlib import Path

# Add src to path for imports (useful for Colab)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catalogue_biencoder.config import TrainConfig
from catalogue_biencoder.training.runner import run


def main():
    """Main entry point: create config and run training."""
    cfg = TrainConfig()
    run(cfg)


if __name__ == "__main__":
    main()
