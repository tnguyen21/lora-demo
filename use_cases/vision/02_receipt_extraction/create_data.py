"""Generate training data for receipt data extraction via VLM teacher."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config

CONFIG = load_config(__file__)
OUTPUT_PATH = "/tmp/tinker-datasets/receipt_extraction.jsonl"


def main():
    print(f"Receipt data extraction data generation")
    print(f"Config: {CONFIG.name}, output format: {CONFIG.output_format}")
    print(f"Data source: {CONFIG.data_source}")
    print()
    print("To generate training data with real receipt images:")
    print("  1. Download images from SROIE or another receipt dataset")
    print("  2. Pass image paths to shared.teacher.generate_teacher_labels_vlm()")
    print()
    print("For the workshop demo, use the pre-generated sample_data/train_sample.jsonl")


if __name__ == "__main__":
    main()
