"""JSONL I/O, train/test split, and synthetic input generation."""

import json
import os
import random
import re
from pathlib import Path

from shared.config import UseCaseConfig
from shared.output_parsers import parse_json_output, parse_single_label, parse_sql


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


def determine_test_size(
    total_records: int,
    eval_samples: int | None = None,
    default: float = 0.2,
    max_test_size: float = 0.5,
) -> float:
    """Pick a test split size based on eval needs and dataset size."""
    if total_records <= 0:
        return default
    if not eval_samples or eval_samples <= 0:
        return default
    desired = eval_samples / total_records
    return min(max_test_size, max(default, desired))


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


def looks_like_repeated_answer(text: str, min_repeats: int = 5) -> bool:
    """Heuristic for degenerate outputs like repeated 'Answer:' lines."""
    if not text:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    normalized = [re.sub(r"\s+", " ", line).strip().lower() for line in lines]
    answer_count = sum(1 for line in normalized if line.rstrip(":") == "answer")
    if answer_count >= min_repeats:
        return True

    max_consecutive = 1
    current = 1
    for i in range(1, len(normalized)):
        if normalized[i] == normalized[i - 1]:
            current += 1
            if current > max_consecutive:
                max_consecutive = current
        else:
            current = 1
    if max_consecutive >= min_repeats:
        return True

    if len(normalized) >= 10:
        unique_ratio = len(set(normalized)) / len(normalized)
        if unique_ratio < 0.2:
            return True

    return False


def filter_degenerate_records(records: list[dict], min_repeats: int = 5) -> tuple[list[dict], int]:
    """Filter message records with degenerate assistant outputs."""
    cleaned = []
    removed = 0
    for record in records:
        assistant_content = record["messages"][1]["content"]
        if looks_like_repeated_answer(assistant_content, min_repeats=min_repeats):
            removed += 1
            continue
        cleaned.append(record)
    return cleaned, removed


def _normalize_label_map(labels: list[str]) -> dict[str, str]:
    return {label.lower(): label.lower() for label in labels}


def filter_single_label_records(
    records: list[dict],
    labels: list[str],
    output_regex: str | None = None,
) -> tuple[list[dict], int]:
    """Keep only records with valid single-label outputs."""
    normalized_labels = _normalize_label_map(labels)
    cleaned = []
    removed = 0
    for record in records:
        content = record["messages"][1]["content"]
        parsed = parse_single_label(content, labels, regex=output_regex)
        if not parsed:
            removed += 1
            continue
        record["messages"][1]["content"] = normalized_labels[parsed.lower()]
        cleaned.append(record)
    return cleaned, removed


def _extract_sql(text: str) -> str | None:
    parsed = parse_sql(text)
    if parsed:
        return parsed.strip()
    match = re.search(r"(?is)\b(select|with|insert|update|delete)\b.*", text)
    if match:
        return match.group(0).strip()
    return None


def filter_sql_generation_records(records: list[dict]) -> tuple[list[dict], int]:
    """Keep only records with plausible SQL outputs, normalized to a single statement."""
    cleaned = []
    removed = 0
    for record in records:
        content = record["messages"][1]["content"]
        if looks_like_repeated_answer(content):
            removed += 1
            continue
        sql = _extract_sql(content)
        if not sql:
            removed += 1
            continue
        sql = re.sub(r"(?is)^.*?\b(select|with|insert|update|delete)\b", r"\1", sql).strip()
        if not sql.endswith(";"):
            sql = f"{sql};"
        record["messages"][1]["content"] = sql
        cleaned.append(record)
    return cleaned, removed


def _parse_json_or_none(text: str) -> dict | list | None:
    return parse_json_output(text)


def filter_json_records(records: list[dict]) -> tuple[list[dict], int]:
    """Keep only records with valid JSON outputs."""
    cleaned = []
    removed = 0
    for record in records:
        content = record["messages"][1]["content"]
        parsed = _parse_json_or_none(content)
        if parsed is None:
            removed += 1
            continue
        record["messages"][1]["content"] = json.dumps(parsed, ensure_ascii=True)
        cleaned.append(record)
    return cleaned, removed


def filter_pii_detection_records(records: list[dict], labels: list[str]) -> tuple[list[dict], int]:
    """Validate PII JSON array structure and label types."""
    label_set = {label.lower() for label in labels}
    cleaned = []
    removed = 0
    for record in records:
        content = record["messages"][1]["content"]
        parsed = _parse_json_or_none(content)
        if parsed is None:
            removed += 1
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
            parsed = parsed["entities"]
        if not isinstance(parsed, list):
            removed += 1
            continue
        entities = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            entity = item.get("entity")
            ent_type = item.get("type")
            if not isinstance(entity, str) or not entity.strip():
                continue
            if not isinstance(ent_type, str) or ent_type.lower() not in label_set:
                continue
            entities.append({"entity": entity.strip(), "type": ent_type.lower()})
        record["messages"][1]["content"] = json.dumps(entities, ensure_ascii=True)
        cleaned.append(record)
    return cleaned, removed


def filter_sentiment_aspect_records(records: list[dict], labels: list[str]) -> tuple[list[dict], int]:
    """Validate sentiment/aspect JSON schema."""
    label_set = {label.lower() for label in labels}
    cleaned = []
    removed = 0
    for record in records:
        content = record["messages"][1]["content"]
        parsed = _parse_json_or_none(content)
        if not isinstance(parsed, dict):
            removed += 1
            continue
        sentiment = parsed.get("sentiment")
        aspect = parsed.get("aspect")
        confidence = parsed.get("confidence")
        reasoning = parsed.get("reasoning")
        if not isinstance(sentiment, str) or sentiment.lower() not in label_set:
            removed += 1
            continue
        if not isinstance(aspect, str) or not aspect.strip():
            removed += 1
            continue
        if not isinstance(reasoning, str) or not reasoning.strip():
            removed += 1
            continue
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            removed += 1
            continue
        normalized_aspect = re.sub(r"\s+", "_", aspect.strip().lower())
        normalized = {
            "sentiment": sentiment.lower(),
            "aspect": normalized_aspect,
            "confidence": float(confidence),
            "reasoning": reasoning.strip(),
        }
        record["messages"][1]["content"] = json.dumps(normalized, ensure_ascii=True)
        cleaned.append(record)
    return cleaned, removed


def filter_receipt_extraction_records(records: list[dict]) -> tuple[list[dict], int]:
    """Validate receipt extraction JSON schema."""
    cleaned = []
    removed = 0
    for record in records:
        content = record["messages"][1]["content"]
        parsed = _parse_json_or_none(content)
        if not isinstance(parsed, dict):
            removed += 1
            continue
        if not {"vendor", "date", "total", "line_items"}.issubset(parsed.keys()):
            removed += 1
            continue
        vendor = parsed.get("vendor")
        date = parsed.get("date")
        total = parsed.get("total")
        line_items = parsed.get("line_items")
        if vendor is not None and not isinstance(vendor, str):
            removed += 1
            continue
        if date is not None and not isinstance(date, str):
            removed += 1
            continue
        if total is not None and not isinstance(total, (int, float)):
            removed += 1
            continue
        if not isinstance(line_items, list):
            removed += 1
            continue
        cleaned_items = []
        for item in line_items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            price = item.get("price")
            if not isinstance(name, str) or not name.strip():
                continue
            if price is not None and not isinstance(price, (int, float)):
                continue
            cleaned_items.append({"name": name.strip(), "price": price})
        normalized = {
            "vendor": vendor.strip() if isinstance(vendor, str) else None,
            "date": date.strip() if isinstance(date, str) else None,
            "total": float(total) if isinstance(total, (int, float)) else None,
            "line_items": cleaned_items,
        }
        record["messages"][1]["content"] = json.dumps(normalized, ensure_ascii=True)
        cleaned.append(record)
    return cleaned, removed


def filter_records_for_use_case(
    config: UseCaseConfig,
    records: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Apply general + use-case-specific filters."""
    removed_by: dict[str, int] = {}

    records, removed = filter_degenerate_records(records)
    if removed:
        removed_by["degenerate"] = removed

    if config.output_format == "single_label":
        records, removed = filter_single_label_records(records, config.labels, output_regex=config.output_regex)
        if removed:
            removed_by["single_label"] = removed
    elif config.output_format == "json":
        if config.name == "pii_detection":
            records, removed = filter_pii_detection_records(records, config.labels)
            if removed:
                removed_by["pii_schema"] = removed
        elif config.name == "sentiment_aspect":
            records, removed = filter_sentiment_aspect_records(records, config.labels)
            if removed:
                removed_by["sentiment_schema"] = removed
        elif config.name == "receipt_extraction":
            records, removed = filter_receipt_extraction_records(records)
            if removed:
                removed_by["receipt_schema"] = removed
        else:
            records, removed = filter_json_records(records)
            if removed:
                removed_by["json"] = removed
    elif config.output_format == "free_text" and config.name == "sql_generation":
        records, removed = filter_sql_generation_records(records)
        if removed:
            removed_by["sql"] = removed

    return records, removed_by
