"""Evaluate 3 models side-by-side for PII detection."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config
from shared.eval_harness import run_comparison_eval

CONFIG = load_config(__file__)
TEST_DATA_PATH = "/tmp/tinker-datasets/pii_detection_test.jsonl"


def ensure_test_data() -> None:
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Missing test data at {TEST_DATA_PATH}. Run create_data.py to generate train/test splits.")


def main(checkpoint_path: str | None = None):
    ensure_test_data()
    return asyncio.run(
        run_comparison_eval(
            config=CONFIG,
            test_data_path=TEST_DATA_PATH,
            checkpoint_path=checkpoint_path,
            max_eval=CONFIG.eval_samples,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Tinker checkpoint path for fine-tuned model")
    args = parser.parse_args()
    main(args.checkpoint)
