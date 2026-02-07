"""Generate training data for document type classification via VLM teacher."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config
from shared.teacher import generate_teacher_labels_vlm
from shared.data_utils import load_jsonl, save_jsonl, train_test_split

CONFIG = load_config(__file__)
OUTPUT_PATH = "/tmp/tinker-datasets/document_type.jsonl"
TEST_OUTPUT_PATH = "/tmp/tinker-datasets/document_type_test.jsonl"
TEST_SPLIT = 0.2
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def collect_images(image_dir: Path) -> list[str]:
    return sorted(str(path) for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing document images for labeling",
    )
    args = parser.parse_args()

    if not args.image_dir:
        print("Document type classification data generation")
        print(f"Config: {CONFIG.name}, labels: {CONFIG.labels}")
        print(f"Data source: {CONFIG.data_source}")
        print()
        print("To generate training data with real images:")
        print("  1. Download images from RVL-CDIP or another document dataset")
        print("  2. Run this script with --image-dir /path/to/images")
        print()
        print("For the workshop demo, use the pre-generated sample_data/train_sample.jsonl")
        return

    image_dir = Path(os.path.expanduser(args.image_dir))
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = collect_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    print(f"Labeling {len(image_paths)} images with {CONFIG.teacher_model}")
    import asyncio

    asyncio.run(generate_teacher_labels_vlm(CONFIG, image_paths, OUTPUT_PATH))
    records = load_jsonl(OUTPUT_PATH)
    train_records, test_records = train_test_split(records, test_size=TEST_SPLIT)
    save_jsonl(train_records, OUTPUT_PATH)
    save_jsonl(test_records, TEST_OUTPUT_PATH)
    print(f"Saved {len(train_records)} train examples to {OUTPUT_PATH}")
    print(f"Saved {len(test_records)} test examples to {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
