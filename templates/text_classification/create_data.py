"""Generate training data — copy and customize for your use case."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.import_utils import load_config
from shared.teacher import generate_teacher_labels
from shared.data_utils import generate_synthetic_inputs

CONFIG = load_config(__file__)
OUTPUT_PATH = "/tmp/tinker-datasets/my_classifier.jsonl"  # TODO: Change output path


def main():
    inputs = generate_synthetic_inputs(CONFIG)
    print(f"Generated {len(inputs)} synthetic inputs")
    asyncio.run(generate_teacher_labels(CONFIG, inputs, OUTPUT_PATH))


if __name__ == "__main__":
    main()
