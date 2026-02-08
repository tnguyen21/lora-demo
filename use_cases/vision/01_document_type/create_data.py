"""Generate training data for document type classification via VLM teacher.

For the vision use case, this script loads images from a HuggingFace dataset
(RVL-CDIP subset) and uses a VLM teacher to label them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config

CONFIG = load_config(__file__)
TRAIN_OUTPUT_PATH = "/tmp/tinker-datasets/document_type.jsonl"
TEST_OUTPUT_PATH = "/tmp/tinker-datasets/document_type_test.jsonl"


def main():
    # For vision, we'd typically load images from a HF dataset.
    # This script demonstrates the pattern — in practice, replace with actual image paths.
    print("Document type classification data generation")
    print(f"Config: {CONFIG.name}, labels: {CONFIG.labels}")
    print(f"Data source: {CONFIG.data_source}")

    print()
    print("To generate training data with real images:")
    print("  1. Download images from RVL-CDIP or another document dataset")
    print("  2. Pass image paths to shared.teacher.generate_teacher_labels_vlm()")


if __name__ == "__main__":
    main()
