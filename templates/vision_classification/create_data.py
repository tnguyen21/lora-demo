"""Generate training data for vision classification — copy and customize."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.import_utils import load_config

CONFIG = load_config(__file__)


def main():
    print(f"Vision classification data generation for: {CONFIG.name}")
    print(f"Labels: {CONFIG.labels}")
    print()
    print("To generate training data:")
    print("  1. Collect or download images for each class")
    print("  2. Pass image paths to shared.teacher.generate_teacher_labels_vlm()")
    print("  3. Or manually create JSONL with image path -> label pairs")


if __name__ == "__main__":
    main()
