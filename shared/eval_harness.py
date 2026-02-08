"""3-model comparison evaluator.

Evaluates held-out test data across three model tiers:
1. Strong API model — teacher model with full prompt
2. Smaller/cheaper API model — same full prompt, smaller model
3. Fine-tuned student — LoRA checkpoint, raw input only (no prompt)

Follows the SamplingClientEvaluator pattern from vlm_classifier/eval.py.
"""

import asyncio
import json
import re

import tinker
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from shared.config import UseCaseConfig
from shared.data_utils import load_jsonl
from shared.output_parsers import parse_json_output, parse_single_label, parse_sql

try:
    import sqlglot  # type: ignore

    _HAS_SQLGLOT = True
except Exception:
    sqlglot = None
    _HAS_SQLGLOT = False


COMPARISON_MODELS = {
    "strong_api": {
        "display_name": "Teacher (Qwen3-30B-A3B + full prompt)",
        "model": "Qwen/Qwen3-30B-A3B",
        "renderer": "qwen3",
        "uses_teacher_prompt": True,
    },
}

SMALLER_API_MODEL = {
    "display_name": "Smaller API (Qwen3-30B-A3B + full prompt)",
    "model": "Qwen/Qwen3-30B-A3B",
    "renderer": "qwen3",
    "uses_teacher_prompt": True,
}


def _normalize_sql(text: str) -> str | None:
    parsed = parse_sql(text)
    if not parsed:
        return None
    cleaned = parsed.strip().rstrip(";")
    if _HAS_SQLGLOT:
        try:
            expression = sqlglot.parse_one(cleaned, read="postgres")
            cleaned = expression.sql(dialect="postgres", pretty=False)
        except Exception:
            pass
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


async def _evaluate_model(
    config: UseCaseConfig,
    test_data: list[dict],
    model_name: str,
    renderer_name: str,
    uses_teacher_prompt: bool,
    model_path: str | None = None,
    max_parallel: int = 64,
) -> dict[str, float]:
    """Evaluate a single model on test data."""
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    service_client = tinker.ServiceClient()
    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    params = tinker.SamplingParams(
        max_tokens=config.teacher_max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )

    semaphore = asyncio.Semaphore(max_parallel)

    async def eval_one(example: dict) -> dict[str, float]:
        async with semaphore:
            user_input = example["messages"][0]["content"]
            expected = example["messages"][1]["content"]

            if uses_teacher_prompt:
                prompt_text = config.teacher_prompt.format(input=user_input)
                tokenized = tinker.ModelInput.from_ints(tokenizer.encode(prompt_text))
            else:
                messages = [{"role": "user", "content": user_input}]
                tokenized = renderer.build_generation_prompt(messages)

            result = await sampling_client.sample_async(prompt=tokenized, sampling_params=params, num_samples=1)
            response = tokenizer.decode(result.sequences[0].tokens)

            if config.output_format == "single_label":
                predicted = parse_single_label(response, config.labels, config.output_regex)
                expected_parsed = expected.strip().lower()
                correct = float(predicted == expected_parsed) if predicted else 0.0
                return {"accuracy": correct}
            elif config.output_format == "json":
                predicted = parse_json_output(response)
                expected_parsed = json.loads(expected)
                correct = float(predicted == expected_parsed) if predicted else 0.0
                return {"accuracy": correct}
            else:
                # free_text: exact match, SQL gets normalized comparison
                if config.name == "sql_generation":
                    predicted = _normalize_sql(response)
                    expected_norm = _normalize_sql(expected)
                    correct = float(predicted == expected_norm) if predicted and expected_norm else 0.0
                    return {"accuracy": correct}
                predicted = response.strip()
                correct = float(predicted.lower() == expected.strip().lower())
                return {"accuracy": correct}

    tasks = [asyncio.create_task(eval_one(ex)) for ex in test_data]
    results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        result = await coro
        results.append(result)

    if not results:
        return {"accuracy": 0.0}

    avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
    return {"accuracy": avg_accuracy, "n_examples": float(len(results))}


async def run_comparison_eval(
    config: UseCaseConfig,
    test_data_path: str,
    checkpoint_path: str | None = None,
    max_eval: int | None = None,
) -> dict[str, dict[str, float]]:
    """Run 3-model comparison evaluation.

    Args:
        config: Use case config.
        test_data_path: Path to test JSONL file.
        checkpoint_path: Tinker checkpoint path for the fine-tuned model.
        max_eval: Maximum number of examples to evaluate (None = all).

    Returns:
        Dict mapping model name to metrics dict.
    """
    test_data = load_jsonl(test_data_path)
    if max_eval:
        test_data = test_data[:max_eval]

    results = {}

    # 1. Teacher model with full prompt
    print(f"\nEvaluating: Teacher ({config.teacher_model} + full prompt)")
    teacher_metrics = await _evaluate_model(
        config=config,
        test_data=test_data,
        model_name=config.teacher_model,
        renderer_name=config.renderer_name,
        uses_teacher_prompt=True,
    )
    results["teacher"] = teacher_metrics
    print(f"  Accuracy: {teacher_metrics['accuracy']:.3f}")

    # 2. Smaller API model with full prompt
    print(f"\nEvaluating: Smaller API ({config.teacher_model} + full prompt)")
    smaller_metrics = await _evaluate_model(
        config=config,
        test_data=test_data,
        model_name=config.teacher_model,
        renderer_name=config.renderer_name,
        uses_teacher_prompt=True,
    )
    results["smaller_api"] = smaller_metrics
    print(f"  Accuracy: {smaller_metrics['accuracy']:.3f}")

    # 3. Fine-tuned student (no prompt)
    if checkpoint_path:
        print("\nEvaluating: Fine-tuned student (checkpoint, raw input only)")
        student_metrics = await _evaluate_model(
            config=config,
            test_data=test_data,
            model_name=config.student_model,
            renderer_name=config.renderer_name,
            uses_teacher_prompt=False,
            model_path=checkpoint_path,
        )
        results["fine_tuned"] = student_metrics
        print(f"  Accuracy: {student_metrics['accuracy']:.3f}")
    else:
        print("\nSkipping fine-tuned model eval (no checkpoint_path provided)")

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Model':<40s} {'Accuracy':>10s}")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:<40s} {metrics['accuracy']:>10.3f}")
    print("=" * 60)

    return results
