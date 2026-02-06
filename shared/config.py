"""Central configuration schema for all use cases."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class UseCaseConfig:
    """Configuration for a text use case."""

    name: str
    display_name: str
    labels: list[str]
    teacher_prompt: str  # full rubric with {input} placeholder
    output_format: Literal["single_label", "json", "free_text"]
    category: Literal["text", "vision"] = "text"
    output_regex: str | None = None  # regex to parse teacher output

    # model settings
    teacher_model: str = "Qwen/Qwen3-30B-A3B"
    student_model: str = "Qwen/Qwen3-30B-A3B"
    renderer_name: str = "qwen3"

    # training defaults
    learning_rate: float = 1e-4
    num_epochs: int = 4
    lora_rank: int = 32
    batch_size: int = 128
    max_length: int = 32768
    lr_schedule: str = "linear"

    # token estimates (for cost calc)
    teacher_input_tokens: int = 600
    student_input_tokens: int = 30
    teacher_output_tokens: int = 4
    student_output_tokens: int = 1

    # data
    synthetic_examples: int = 1000
    eval_samples: int = 200

    # synthetic input templates for data generation
    synthetic_input_templates: list[str] = field(default_factory=list)

    # teacher sampling params
    teacher_temperature: float = 0.15
    teacher_max_tokens: int = 1000


@dataclass
class VisionUseCaseConfig(UseCaseConfig):
    """Configuration for a vision use case."""

    category: Literal["text", "vision"] = "vision"
    teacher_model: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    student_model: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    renderer_name: str = "qwen3_vl"
    max_image_size: int = 480
    data_source: str | None = None  # HF dataset name
