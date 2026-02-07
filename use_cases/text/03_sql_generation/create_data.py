"""Generate training data for SQL generation via teacher model."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config
from shared.teacher import generate_teacher_labels
from shared.data_utils import generate_synthetic_inputs, load_jsonl, save_jsonl, train_test_split

CONFIG = load_config(__file__)
OUTPUT_PATH = "/tmp/tinker-datasets/sql_generation.jsonl"
TEST_OUTPUT_PATH = "/tmp/tinker-datasets/sql_generation_test.jsonl"
TEST_SPLIT = 0.2


def main():
    inputs = generate_synthetic_inputs(CONFIG)
    print(f"Generated {len(inputs)} synthetic SQL generation inputs")
    asyncio.run(generate_teacher_labels(CONFIG, inputs, OUTPUT_PATH))
    records = load_jsonl(OUTPUT_PATH)
    train_records, test_records = train_test_split(records, test_size=TEST_SPLIT)
    save_jsonl(train_records, OUTPUT_PATH)
    save_jsonl(test_records, TEST_OUTPUT_PATH)
    print(f"Saved {len(train_records)} train examples to {OUTPUT_PATH}")
    print(f"Saved {len(test_records)} test examples to {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
