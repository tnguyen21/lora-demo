"""Parameterized cost comparison calculator.

Generalizes scripts/cost_comparison.py to accept any UseCaseConfig's token estimates.
"""

from shared.config import UseCaseConfig

# Pricing per 1M tokens (USD) — February 2026
MODELS = {
    "GPT-5.2 (frontier)": {
        "input_per_1m": 1.75,
        "output_per_1m": 14.00,
    },
    "GPT-4.1-mini (small)": {
        "input_per_1m": 0.40,
        "output_per_1m": 1.60,
    },
    "Gemini 2.5 Flash (small)": {
        "input_per_1m": 0.30,
        "output_per_1m": 2.50,
    },
    "Claude Haiku 4.5 (small)": {
        "input_per_1m": 1.00,
        "output_per_1m": 5.00,
    },
    "Tinker LoRA (self-hosted)": {
        "input_per_1m": 0.10,
        "output_per_1m": 0.10,
    },
}

VOLUME_TIERS = [1_000, 10_000, 100_000, 1_000_000]


def cost_per_request(input_tokens: float, output_tokens: float, model: dict) -> float:
    return (input_tokens * model["input_per_1m"] + output_tokens * model["output_per_1m"]) / 1_000_000


def format_usd(amount: float) -> str:
    if amount < 0.001:
        return f"${amount:.6f}"
    if amount < 0.01:
        return f"${amount:.4f}"
    if amount < 1.0:
        return f"${amount:.3f}"
    if amount < 100:
        return f"${amount:.2f}"
    return f"${amount:,.0f}"


def format_k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def compute_cost_comparison(config: UseCaseConfig) -> str:
    """Compute and return a formatted cost comparison report for a use case.

    Uses the token estimates from the config to compute costs across
    multiple model/pricing tiers and volume levels.

    Returns:
        Formatted report string.
    """
    lines = []

    lines.append("=" * 72)
    lines.append(f"{config.display_name.upper()}: TOKEN & COST COMPARISON")
    lines.append("=" * 72)

    # Token counts
    lines.append("\n## Token Counts Per Request")
    lines.append(f"{'':40s} {'Input':>10s} {'Output':>10s}")
    lines.append(f"{'-' * 40} {'-' * 10} {'-' * 10}")
    lines.append(
        f"{'Teacher prompt (API models)':40s} {'~' + str(config.teacher_input_tokens):>10s} {'~' + str(config.teacher_output_tokens):>10s}"
    )
    lines.append(
        f"{'Distilled (fine-tuned, no prompt)':40s} "
        f"{'~' + str(config.student_input_tokens):>10s} "
        f"{'~' + str(config.student_output_tokens):>10s}"
    )

    if config.student_input_tokens > 0:
        reduction = config.teacher_input_tokens / config.student_input_tokens
        lines.append(f"\n  -> Input token reduction: {reduction:.0f}x")

    # Per-request cost
    lines.append("\n## Cost Per Request")
    header = f"{'Model':36s} {'Approach':12s} {'Input tok':>10s} {'Output tok':>10s} {'Cost/req':>12s}"
    lines.append(header)
    lines.append("-" * len(header))

    rows = []
    for name, pricing in MODELS.items():
        is_tinker = "Tinker" in name
        input_tok = config.student_input_tokens if is_tinker else config.teacher_input_tokens
        output_tok = config.student_output_tokens if is_tinker else config.teacher_output_tokens
        approach = "distilled" if is_tinker else "teacher"
        cpr = cost_per_request(input_tok, output_tok, pricing)
        rows.append((name, approach, input_tok, output_tok, cpr))
        lines.append(f"{name:36s} {approach:12s} {input_tok:>10d} {output_tok:>10d} {format_usd(cpr):>12s}")

    # Daily cost at volume
    lines.append("\n## Daily Cost by Volume (requests/day)")
    vol_header = f"{'Model':36s}" + "".join(f" {format_k(v) + '/day':>12s}" for v in VOLUME_TIERS)
    lines.append(vol_header)
    lines.append("-" * len(vol_header))

    for name, approach, input_tok, output_tok, cpr in rows:
        line = f"{name:36s}"
        for vol in VOLUME_TIERS:
            daily = cpr * vol
            line += f" {format_usd(daily):>12s}"
        lines.append(line)

    # Monthly cost at volume
    lines.append("\n## Monthly Cost (30 days)")
    lines.append(vol_header)
    lines.append("-" * len(vol_header))

    for name, approach, input_tok, output_tok, cpr in rows:
        line = f"{name:36s}"
        for vol in VOLUME_TIERS:
            monthly = cpr * vol * 30
            line += f" {format_usd(monthly):>12s}"
        lines.append(line)

    # Savings summary
    frontier_cpr = rows[0][4]
    tinker_cpr = rows[-1][4]
    best_small_cpr = min(r[4] for r in rows[1:-1])
    best_small_name = [r[0] for r in rows[1:-1] if r[4] == best_small_cpr][0]

    lines.append(f"\n## Savings vs Frontier ({rows[0][0]})")
    lines.append(f"  Best small API model ({best_small_name}):")
    if best_small_cpr > 0:
        lines.append(f"    {frontier_cpr / best_small_cpr:.1f}x cheaper per request")
    lines.append(f"    At 100K req/day: save {format_usd((frontier_cpr - best_small_cpr) * 100_000)}/day")
    lines.append("\n  Fine-tuned on Tinker:")
    if tinker_cpr > 0:
        lines.append(f"    {frontier_cpr / tinker_cpr:.0f}x cheaper per request")
    lines.append(f"    At 100K req/day: save {format_usd((frontier_cpr - tinker_cpr) * 100_000)}/day")
    lines.append("")

    return "\n".join(lines)
