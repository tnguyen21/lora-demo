"""Generate training data for PII detection via teacher model."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config
from shared.teacher import generate_teacher_labels
from shared.data_utils import (
    determine_test_size,
    filter_records_for_use_case,
    generate_synthetic_inputs,
    messages_to_jsonl_record,
    save_jsonl,
    train_test_split,
)

CONFIG = load_config(__file__)
TRAIN_OUTPUT_PATH = "/tmp/tinker-datasets/pii_detection.jsonl"
TEST_OUTPUT_PATH = "/tmp/tinker-datasets/pii_detection_test.jsonl"


def main():
    inputs = generate_synthetic_inputs(CONFIG)
    print(f"Generated {len(inputs)} synthetic inputs")
    results = asyncio.run(generate_teacher_labels(CONFIG, inputs, TRAIN_OUTPUT_PATH))
    records = [messages_to_jsonl_record(r["input"], r["output"]) for r in results]
    records, removed_by = filter_records_for_use_case(CONFIG, records)
    if removed_by:
        total_removed = sum(removed_by.values())
        details = ", ".join(f"{reason}={count}" for reason, count in removed_by.items())
        print(f"Filtered {total_removed} records ({details})")
    test_size = determine_test_size(len(records), eval_samples=CONFIG.eval_samples)
    train_records, test_records = train_test_split(records, test_size=test_size, seed=42)
    save_jsonl(train_records, TRAIN_OUTPUT_PATH)
    save_jsonl(test_records, TEST_OUTPUT_PATH)
    print(
        f"Split {len(records)} records into {len(train_records)} train / {len(test_records)} test"
        f" (test_size={test_size:.2f})"
    )
    print(f"Saved train data to {TRAIN_OUTPUT_PATH}")
    print(f"Saved test data to {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
