"""JSONL I/O, train/test split, and synthetic input generation."""

import json
import os
import random
from pathlib import Path

from shared.config import UseCaseConfig


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Save a list of dicts as a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def train_test_split(
    records: list[dict],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Deterministic train/test split."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_size))
    return shuffled[:split_idx], shuffled[split_idx:]


def generate_synthetic_inputs(config: UseCaseConfig, seed: int = 42) -> list[str]:
    """Generate synthetic inputs from config's templates.

    Each template is used roughly equally to produce config.synthetic_examples inputs.
    Templates can contain {idx} for a unique index.
    """
    if not config.synthetic_input_templates:
        raise ValueError(f"No synthetic_input_templates defined for {config.name}")

    rng = random.Random(seed)
    templates = config.synthetic_input_templates
    inputs = []
    for i in range(config.synthetic_examples):
        template = templates[i % len(templates)]
        # Add some variation by shuffling template selection after first pass
        if i >= len(templates):
            template = rng.choice(templates)
        inputs.append(template.format(idx=i))
    return inputs


def messages_to_jsonl_record(user_content: str, assistant_content: str) -> dict:
    """Create a JSONL record in the short message format for training."""
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
