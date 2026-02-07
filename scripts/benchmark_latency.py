"""Benchmark inference latency for teacher vs fine-tuned student models.

Measures wall-clock time per request using sequential requests for clean
per-request latency numbers. Reports p50, p95, and mean latency.

Requires a Tinker API connection. Without a checkpoint, benchmarks teacher only.

Usage:
    python scripts/benchmark_latency.py use_cases/text/01_ticket_routing
    python scripts/benchmark_latency.py use_cases/text/01_ticket_routing --checkpoint tinker://<run-id>/sampler_weights/final
    python scripts/benchmark_latency.py use_cases/text/01_ticket_routing -n 10
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from shared.config import UseCaseConfig
from shared.data_utils import load_jsonl


def benchmark_model(
    config: UseCaseConfig,
    samples: list[dict],
    sampling_client: tinker.SamplingClient,
    tokenizer,
    renderer,
    uses_teacher_prompt: bool,
) -> list[float]:
    """Run sequential requests and return per-request latencies in ms."""
    params = tinker.SamplingParams(
        max_tokens=config.teacher_max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )

    latencies: list[float] = []
    for i, example in enumerate(samples):
        user_input = example["messages"][0]["content"]

        if uses_teacher_prompt:
            prompt_text = config.teacher_prompt.format(input=user_input)
            tokenized = tinker.ModelInput.from_ints(tokenizer.encode(prompt_text))
        else:
            messages = [{"role": "user", "content": user_input}]
            tokenized = renderer.build_generation_prompt(messages)

        start = time.perf_counter()
        sampling_client.sample(prompt=tokenized, sampling_params=params, num_samples=1).result()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        print(f"  [{i + 1}/{len(samples)}] {elapsed_ms:.1f}ms")

    return latencies


def print_stats(label: str, latencies: list[float]):
    """Print p50, p95, mean for a list of latencies."""
    if not latencies:
        print(f"\n{label}: no data")
        return
    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]
    mean = statistics.mean(latencies_sorted)
    print(f"\n{label}:")
    print(f"  p50:  {p50:>8.1f}ms")
    print(f"  p95:  {p95:>8.1f}ms")
    print(f"  mean: {mean:>8.1f}ms")
    return p50


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference latency for teacher vs student models.")
    parser.add_argument("use_case_dir", type=str, help="Path to use case directory (e.g. use_cases/text/01_ticket_routing)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Tinker checkpoint path for the fine-tuned student model")
    parser.add_argument("-n", "--num-requests", type=int, default=20, help="Number of requests to benchmark (default: 20)")
    args = parser.parse_args()

    # Load config
    repo_root = Path(__file__).resolve().parents[1]
    use_case_path = repo_root / args.use_case_dir
    config_path = use_case_path / "config.py"

    if not config_path.exists():
        print(f"Error: config.py not found at {config_path}")
        sys.exit(1)

    import importlib.util

    spec = importlib.util.spec_from_file_location("_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config: UseCaseConfig = mod.CONFIG

    # Load sample data
    sample_data_path = use_case_path / "sample_data" / "train_sample.jsonl"
    if not sample_data_path.exists():
        print(f"Error: sample data not found at {sample_data_path}")
        sys.exit(1)

    all_samples = load_jsonl(str(sample_data_path))
    samples = all_samples[: args.num_requests]
    print(f"Benchmarking {len(samples)} requests for: {config.display_name}")

    # Setup Tinker client
    service_client = tinker.ServiceClient()
    tokenizer = get_tokenizer(config.teacher_model)
    renderer = renderers.get_renderer(config.renderer_name, tokenizer)

    # Benchmark teacher
    print(f"\n--- Teacher ({config.teacher_model} + full prompt) ---")
    teacher_client = service_client.create_sampling_client(base_model=config.teacher_model)
    teacher_latencies = benchmark_model(config, samples, teacher_client, tokenizer, renderer, uses_teacher_prompt=True)
    teacher_p50 = print_stats("Teacher", teacher_latencies)

    # Benchmark student (if checkpoint provided)
    student_p50 = None
    if args.checkpoint:
        print("\n--- Student (checkpoint, no prompt) ---")
        student_client = service_client.create_sampling_client(model_path=args.checkpoint)
        student_latencies = benchmark_model(config, samples, student_client, tokenizer, renderer, uses_teacher_prompt=False)
        student_p50 = print_stats("Fine-tuned student", student_latencies)
    else:
        print("\nSkipping student benchmark (no --checkpoint provided)")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if teacher_p50 is not None:
        print(f"Teacher p50:         ~{teacher_p50:.0f}ms")
    if student_p50 is not None:
        print(f"Student p50:         ~{student_p50:.0f}ms")
        if teacher_p50 and student_p50:
            speedup = teacher_p50 / student_p50
            print(f"Speedup:             ~{speedup:.1f}x")


if __name__ == "__main__":
    main()
