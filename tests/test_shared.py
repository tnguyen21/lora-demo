"""Unit tests for shared modules (no API access needed)."""

import os
import tempfile

from shared.config import UseCaseConfig, VisionUseCaseConfig
from shared.cost import compute_cost_comparison, cost_per_request, format_k, format_usd
from shared.data_utils import load_jsonl, messages_to_jsonl_record, save_jsonl, train_test_split
from shared.output_parsers import parse_free_text, parse_json_output, parse_single_label, parse_sql


# ---- config.py ----


class TestConfig:
    def test_use_case_config_defaults(self):
        cfg = UseCaseConfig(
            name="test",
            display_name="Test",
            category="text",
            labels=["a", "b"],
            teacher_prompt="Classify: {input}",
            output_format="single_label",
        )
        assert cfg.teacher_model == "Qwen/Qwen3-30B-A3B"
        assert cfg.learning_rate == 1e-4
        assert cfg.num_epochs == 4
        assert cfg.lora_rank == 32
        assert cfg.batch_size == 128
        assert cfg.teacher_latency_ms is None
        assert cfg.student_latency_ms is None

    def test_latency_fields(self):
        cfg = UseCaseConfig(
            name="test",
            display_name="Test",
            labels=["a", "b"],
            teacher_prompt="Classify: {input}",
            output_format="single_label",
            teacher_latency_ms=450.0,
            student_latency_ms=35.0,
        )
        assert cfg.teacher_latency_ms == 450.0
        assert cfg.student_latency_ms == 35.0

    def test_vision_config_defaults(self):
        cfg = VisionUseCaseConfig(
            name="test_vision",
            display_name="Test Vision",
            labels=["x", "y"],
            teacher_prompt="Classify image: {input}",
            output_format="single_label",
        )
        assert cfg.category == "vision"
        assert cfg.teacher_model == "Qwen/Qwen3-VL-235B-A22B-Instruct"
        assert cfg.renderer_name == "qwen3_vl"
        assert cfg.max_image_size == 480


# ---- data_utils.py ----


class TestDataUtils:
    def test_jsonl_roundtrip(self):
        records = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            save_jsonl(records, path)
            loaded = load_jsonl(path)
            assert loaded == records
        finally:
            os.unlink(path)

    def test_train_test_split_deterministic(self):
        records = [{"i": i} for i in range(100)]
        train1, test1 = train_test_split(records, test_size=0.2, seed=42)
        train2, test2 = train_test_split(records, test_size=0.2, seed=42)
        assert train1 == train2
        assert test1 == test2
        assert len(train1) == 80
        assert len(test1) == 20

    def test_train_test_split_no_overlap(self):
        records = [{"i": i} for i in range(50)]
        train, test = train_test_split(records, test_size=0.3, seed=0)
        train_ids = {r["i"] for r in train}
        test_ids = {r["i"] for r in test}
        assert train_ids.isdisjoint(test_ids)
        assert len(train_ids) + len(test_ids) == 50

    def test_messages_to_jsonl_record(self):
        record = messages_to_jsonl_record("hello", "world")
        assert record == {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        }


# ---- output_parsers.py ----


class TestOutputParsers:
    def test_parse_single_label_with_regex(self):
        labels = ["billing", "technical", "account", "general"]
        response = "Final Answer: billing"
        result = parse_single_label(response, labels, regex=r"Final Answer:\s*(\w+)")
        assert result == "billing"

    def test_parse_single_label_fallback(self):
        labels = ["billing", "technical", "account", "general"]
        result = parse_single_label("This is a technical issue", labels)
        assert result == "technical"

    def test_parse_single_label_no_match(self):
        labels = ["billing", "technical"]
        result = parse_single_label("nothing here", labels)
        assert result is None

    def test_parse_json_output_direct(self):
        result = parse_json_output('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_output_code_block(self):
        response = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = parse_json_output(response)
        assert result == {"key": "value"}

    def test_parse_json_output_array(self):
        result = parse_json_output('Some text [{"a": 1}, {"a": 2}] more text')
        assert result == [{"a": 1}, {"a": 2}]

    def test_parse_json_output_invalid(self):
        result = parse_json_output("no json here at all")
        assert result is None

    def test_parse_free_text_plain(self):
        result = parse_free_text("SELECT * FROM users;")
        assert result == "SELECT * FROM users;"

    def test_parse_free_text_code_block(self):
        result = parse_free_text("```sql\nSELECT * FROM users;\n```")
        assert result == "SELECT * FROM users;"

    def test_parse_free_text_with_regex(self):
        result = parse_free_text("Answer: hello world", regex=r"Answer:\s*(.*)")
        assert result == "hello world"

    def test_parse_sql_code_block(self):
        result = parse_sql("```sql\nSELECT id FROM orders WHERE total > 100;\n```")
        assert result == "SELECT id FROM orders WHERE total > 100;"


# ---- cost.py ----


class TestCost:
    def test_cost_per_request(self):
        model = {"input_per_1m": 1.0, "output_per_1m": 2.0}
        result = cost_per_request(1_000_000, 1_000_000, model)
        assert result == 3.0

    def test_cost_per_request_small(self):
        model = {"input_per_1m": 0.10, "output_per_1m": 0.10}
        result = cost_per_request(30, 1, model)
        expected = (30 * 0.10 + 1 * 0.10) / 1_000_000
        assert abs(result - expected) < 1e-12

    def test_format_usd(self):
        assert format_usd(0.0001) == "$0.000100"
        assert format_usd(0.005) == "$0.0050"
        assert format_usd(0.50) == "$0.500"
        assert format_usd(42.5) == "$42.50"
        assert format_usd(1234) == "$1,234"

    def test_format_k(self):
        assert format_k(500) == "500"
        assert format_k(1_000) == "1K"
        assert format_k(100_000) == "100K"
        assert format_k(1_000_000) == "1M"

    def test_compute_cost_comparison_runs(self):
        cfg = UseCaseConfig(
            name="test",
            display_name="Test Task",
            category="text",
            labels=["a", "b"],
            teacher_prompt="test {input}",
            output_format="single_label",
            teacher_input_tokens=500,
            student_input_tokens=25,
            teacher_output_tokens=4,
            student_output_tokens=1,
        )
        report = compute_cost_comparison(cfg)
        assert "TEST TASK" in report
        assert "Token Counts" in report
        assert "Cost Per Request" in report
        assert "Daily Cost" in report
        assert "Monthly Cost" in report
        assert "Savings" in report
        # No latency section when fields are None
        assert "Latency Per Request" not in report

    def test_compute_cost_comparison_with_latency(self):
        cfg = UseCaseConfig(
            name="test",
            display_name="Test Task",
            category="text",
            labels=["a", "b"],
            teacher_prompt="test {input}",
            output_format="single_label",
            teacher_input_tokens=500,
            student_input_tokens=25,
            teacher_output_tokens=4,
            student_output_tokens=1,
            teacher_latency_ms=450.0,
            student_latency_ms=35.0,
        )
        report = compute_cost_comparison(cfg)
        assert "Latency Per Request" in report
        assert "~450ms" in report
        assert "~35ms" in report
        assert "12.9x faster" in report

    def test_compute_cost_comparison_teacher_latency_only(self):
        cfg = UseCaseConfig(
            name="test",
            display_name="Test Task",
            category="text",
            labels=["a", "b"],
            teacher_prompt="test {input}",
            output_format="single_label",
            teacher_latency_ms=450.0,
        )
        report = compute_cost_comparison(cfg)
        assert "Latency Per Request" in report
        assert "~450ms" in report
        assert "faster" not in report
