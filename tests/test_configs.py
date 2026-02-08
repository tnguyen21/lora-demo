"""Validate all 6 use case configs load correctly."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

USE_CASES = [
    ("text/01_ticket_routing", "text"),
    ("text/02_pii_detection", "text"),
    ("text/03_sql_generation", "text"),
    ("text/04_sentiment_aspect", "text"),
    ("vision/01_document_type", "vision"),
    ("vision/02_receipt_extraction", "vision"),
]


def _load_config(use_case_dir: str):
    config_path = REPO_ROOT / "use_cases" / use_case_dir / "config.py"
    spec = importlib.util.spec_from_file_location("_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


@pytest.mark.parametrize("use_case_dir,expected_category", USE_CASES)
def test_config_loads(use_case_dir, expected_category):
    config = _load_config(use_case_dir)
    assert config.name, "name should not be empty"
    assert config.display_name, "display_name should not be empty"
    assert config.category == expected_category
    assert len(config.labels) > 0 or config.output_format == "free_text"
    assert "{input}" in config.teacher_prompt
    assert config.output_format in ("single_label", "json", "free_text")


@pytest.mark.parametrize("use_case_dir,expected_category", USE_CASES)
def test_config_has_templates(use_case_dir, expected_category):
    config = _load_config(use_case_dir)
    assert len(config.synthetic_input_templates) >= 20, f"Expected at least 20 templates, got {len(config.synthetic_input_templates)}"


@pytest.mark.parametrize("use_case_dir,expected_category", USE_CASES)
def test_config_token_estimates(use_case_dir, expected_category):
    config = _load_config(use_case_dir)
    assert config.teacher_input_tokens > 0
    assert config.student_input_tokens > 0
    assert config.teacher_output_tokens > 0
    assert config.student_output_tokens > 0
    assert config.teacher_input_tokens > config.student_input_tokens, "Teacher should use more input tokens (includes prompt)"


@pytest.mark.parametrize("use_case_dir,expected_category", USE_CASES)
def test_cost_report_exists(use_case_dir, expected_category):
    cost_report = REPO_ROOT / "use_cases" / use_case_dir / "cost_report.txt"
    assert cost_report.exists(), f"cost_report.txt missing for {use_case_dir}"
