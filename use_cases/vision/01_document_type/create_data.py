"""Generate training data for document type classification via VLM teacher.

For the vision use case, this script loads images from a HuggingFace dataset
(RVL-CDIP subset) and uses a VLM teacher to label them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config
from shared.data_utils import (
    determine_test_size,
    filter_records_for_use_case,
    load_jsonl,
    save_jsonl,
    train_test_split,
)

CONFIG = load_config(__file__)
TRAIN_OUTPUT_PATH = "/tmp/tinker-datasets/document_type.jsonl"
TEST_OUTPUT_PATH = "/tmp/tinker-datasets/document_type_test.jsonl"
SAMPLE_DATA_PATH = Path(__file__).resolve().parent / "sample_data" / "train_sample.jsonl"


def main():
    # For vision, we'd typically load images from a HF dataset.
    # This script demonstrates the pattern — in practice, replace with actual image paths.
    print("Document type classification data generation")
    print(f"Config: {CONFIG.name}, labels: {CONFIG.labels}")
    print(f"Data source: {CONFIG.data_source}")

    if SAMPLE_DATA_PATH.exists():
        records = load_jsonl(SAMPLE_DATA_PATH)
        test_size = determine_test_size(len(records), eval_samples=CONFIG.eval_samples)
        records, removed_by = filter_records_for_use_case(CONFIG, records)
        if removed_by:
            total_removed = sum(removed_by.values())
            details = ", ".join(f"{reason}={count}" for reason, count in removed_by.items())
            print(f"Filtered {total_removed} records ({details})")
        train_records, test_records = train_test_split(records, test_size=test_size, seed=42)
        save_jsonl(train_records, TRAIN_OUTPUT_PATH)
        save_jsonl(test_records, TEST_OUTPUT_PATH)
        print(
            f"Split {len(records)} sample records into {len(train_records)} train / {len(test_records)} test"
            f" (test_size={test_size:.2f})"
        )
        print(f"Saved train data to {TRAIN_OUTPUT_PATH}")
        print(f"Saved test data to {TEST_OUTPUT_PATH}")
    else:
        print("Sample data not found; skipping train/test split.")

    print()
    print("To generate training data with real images:")
    print("  1. Download images from RVL-CDIP or another document dataset")
    print("  2. Pass image paths to shared.teacher.generate_teacher_labels_vlm()")


if __name__ == "__main__":
    main()
