"""Generate training data for sentiment + aspect extraction via teacher model."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config
from shared.teacher import generate_teacher_labels
from shared.data_utils import generate_synthetic_inputs

CONFIG = load_config(__file__)
OUTPUT_PATH = "/tmp/tinker-datasets/sentiment_aspect.jsonl"


def main():
    inputs = generate_synthetic_inputs(CONFIG)
    print(f"Generated {len(inputs)} synthetic review inputs")
    asyncio.run(generate_teacher_labels(CONFIG, inputs, OUTPUT_PATH))


if __name__ == "__main__":
    main()
