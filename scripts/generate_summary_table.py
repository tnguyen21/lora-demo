"""Generate a master comparison table across all 6 use cases.

Runs each use case's cost config and produces a markdown summary table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import UseCaseConfig
from shared.cost import cost_per_request, format_usd, MODELS

USE_CASE_DIRS = [
    "use_cases/text/01_ticket_routing",
    "use_cases/text/02_pii_detection",
    "use_cases/text/03_sql_generation",
    "use_cases/text/04_sentiment_aspect",
    "use_cases/vision/01_document_type",
    "use_cases/vision/02_receipt_extraction",
]


def main():
    repo_root = Path(__file__).resolve().parents[1]

    configs: list[UseCaseConfig] = []
    for d in USE_CASE_DIRS:
        config_path = repo_root / d / "config.py"
        if config_path.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("_cfg", config_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            configs.append(mod.CONFIG)

    # Print summary table
    frontier_pricing = MODELS["GPT-5.2 (frontier)"]
    tinker_pricing = MODELS["Tinker LoRA (self-hosted)"]

    print("# Cost Comparison Summary — All Use Cases")
    print()
    print("| Use Case | Category | Teacher In/Out | Student In/Out | Frontier $/req | Tinker $/req | Savings |")
    print("|---|---|---|---|---|---|---|")

    for cfg in configs:
        frontier_cost = cost_per_request(cfg.teacher_input_tokens, cfg.teacher_output_tokens, frontier_pricing)
        tinker_cost = cost_per_request(cfg.student_input_tokens, cfg.student_output_tokens, tinker_pricing)
        savings = f"{frontier_cost / tinker_cost:.0f}x" if tinker_cost > 0 else "N/A"
        print(
            f"| {cfg.display_name} | {cfg.category} "
            f"| {cfg.teacher_input_tokens}/{cfg.teacher_output_tokens} "
            f"| {cfg.student_input_tokens}/{cfg.student_output_tokens} "
            f"| {format_usd(frontier_cost)} "
            f"| {format_usd(tinker_cost)} "
            f"| {savings} |"
        )

    print()
    print("## Monthly Cost at 100K requests/day")
    print()
    print("| Use Case | GPT-5.2/mo | Tinker/mo | Monthly Savings |")
    print("|---|---|---|---|")

    for cfg in configs:
        frontier_cost = cost_per_request(cfg.teacher_input_tokens, cfg.teacher_output_tokens, frontier_pricing)
        tinker_cost = cost_per_request(cfg.student_input_tokens, cfg.student_output_tokens, tinker_pricing)
        frontier_monthly = frontier_cost * 100_000 * 30
        tinker_monthly = tinker_cost * 100_000 * 30
        savings_monthly = frontier_monthly - tinker_monthly
        print(f"| {cfg.display_name} | {format_usd(frontier_monthly)} | {format_usd(tinker_monthly)} | {format_usd(savings_monthly)} |")


if __name__ == "__main__":
    main()
