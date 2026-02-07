"""Download a Tinker LoRA adapter checkpoint to local disk."""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

import tinker

DEFAULT_OUTPUT_DIR = "/tmp/tinker-weights/ticket_routing"
ARCHIVE_NAME = "adapter.tar"
REQUIRED_FILES = {"adapter_model.safetensors", "adapter_config.json"}


def safe_extract(tar: tarfile.TarFile, output_dir: Path) -> None:
    """Extract a tar file to output_dir, blocking path traversal."""
    output_dir = output_dir.resolve()
    for member in tar.getmembers():
        member_path = output_dir / member.name
        if not str(member_path.resolve()).startswith(str(output_dir)):
            raise RuntimeError(f"Blocked path traversal in tar entry: {member.name}")
    tar.extractall(output_dir)


def download_checkpoint_archive(checkpoint: str, archive_path: Path) -> None:
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    future = rest_client.get_checkpoint_archive_url_from_tinker_path(checkpoint)
    response = future.result()

    print(f"Downloading archive to {archive_path}...")
    urllib.request.urlretrieve(response.url, archive_path)


def list_extracted_files(output_dir: Path) -> list[Path]:
    return sorted(path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Tinker checkpoint path, e.g. tinker://<run-id>/sampler_weights/final",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to extract the adapter (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / ARCHIVE_NAME
    download_checkpoint_archive(args.checkpoint, archive_path)

    print(f"Extracting {archive_path}...")
    with tarfile.open(archive_path, "r:*") as tar:
        safe_extract(tar, output_dir)

    extracted_files = list_extracted_files(output_dir)
    print("Extracted files:")
    for file_path in extracted_files:
        print(f"  {file_path}")

    missing = REQUIRED_FILES - {path.name for path in extracted_files}
    if missing:
        raise FileNotFoundError("Missing expected adapter files: " + ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
