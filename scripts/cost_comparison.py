"""
Token usage and cost comparison for language classification:
  - Frontier model (GPT-5.2) with full teacher prompt
  - Smaller API model (GPT-4.1-mini) with full teacher prompt
  - Fine-tuned model on Tinker (Qwen3-30B-A3B LoRA, distilled — no prompt needed)

Prices as of February 2026.
"""

import tiktoken

# ---------------------------------------------------------------------------
# 1. Define prompts
# ---------------------------------------------------------------------------

TEACHER_PROMPT = """\
You are a precise language classifier.

Goal: Classify the language of the provided text into exactly one of these labels:
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French),
hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese),
zh (Chinese - Simplified), ot (Other/Unknown).

Instructions:
1) Preprocess carefully (without changing the intended meaning):
   - Trim whitespace.
   - Ignore URLs, emails, file paths, hashtags, user handles, and emojis.
   - Ignore numbers, math expressions, and standalone punctuation.
   - If there is code, IGNORE code syntax (keywords, operators, braces) and focus ONLY on human language in comments and string literals.
   - Preserve letters and diacritics; do NOT strip accents.
   - If after ignoring the above there are no alphabetic letters left, output 'ot'.

2) Script-based rules (highest priority):
   - Devanagari script -> hi.
   - Greek script -> el.
   - Cyrillic script -> ru.
   - Han characters -> zh. (Treat Traditional as zh too.)
   - Arabic script -> ar vs ur:
       If Urdu-only letters appear, or clear Urdu words, choose ur.
       Otherwise choose ar.
   (If multiple scripts appear, pick the script that contributes the majority of alphabetic characters.)

3) Latin-script heuristics (use when text is mainly Latin letters):
   - vi: presence of Vietnamese-specific letters/diacritics.
   - tr: presence of Turkish-specific letters and common function words.
   - de: presence of umlauts or sharp-s and common function words.
   - es: presence of enye, inverted punctuation and common words.
   - fr: frequent French diacritics and common words.
   - en: default among Latin languages if strong evidence for others is absent.

4) Named entities & loanwords:
   - Do NOT decide based on a single proper noun, brand, or place name.
   - Require at least two function words or repeated language-specific signals.

5) Mixed-language text:
   - Determine the dominant language by counting indicative tokens.
   - If two or more languages are equally dominant, return 'ot'.

6) Very short or noisy inputs:
   - If the text is 2 or fewer meaningful words, return 'ot' unless there is a strong signal.

7) Transliteration/romanization:
   - If a non-Latin language is written purely in Latin letters without clear cues, return 'ot'.

8) Code-heavy inputs:
   - If the text is mostly code with minimal natural-language, return 'ot'.

9) Ambiguity & confidence:
   - When in doubt, choose 'ot' rather than guessing.

Output format:
- Respond with EXACTLY one line: "Final Answer: xx"
- Where xx is one of: ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot.

Text to classify:
{text}"""

SAMPLE_INPUTS = [
    "And he said, Mama, I'm home.",
    "Et il a dit, maman, je suis à la maison.",
    "Y él dijo: Mamá, estoy en casa.",
    "und er hat gesagt, Mama ich bin daheim.",
    "他说，妈妈，我回来了。",
    "И он сказал: Мама, я дома.",
    "और उसने कहा, माँ, मैं घर आया हूं।",
    "Ve Anne, evdeyim dedi.",
    "Và anh ấy nói, Mẹ, con đã về nhà.",
    "وقال، ماما، لقد عدت للمنزل.",
    "Well, I wasn't even thinking about that, but I was so frustrated, and, I ended up talking to him again.",
    "Eh bien, je ne pensais même pas à cela, mais j'étais si frustré, et j'ai fini par lui reparler.",
    "Nun, daran dachte ich nicht einmal, aber ich war so frustriert, dass ich am Ende doch mit ihm redete.",
]

# ---------------------------------------------------------------------------
# 2. Pricing (per 1M tokens, USD) — February 2026
# ---------------------------------------------------------------------------

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
    "Tinker Qwen3-30B LoRA (self-hosted)": {
        "input_per_1m": 0.10,
        "output_per_1m": 0.10,
    },
}

VOLUME_TIERS = [1_000, 10_000, 100_000, 1_000_000]

# ---------------------------------------------------------------------------
# 3. Count tokens
# ---------------------------------------------------------------------------


def count_tokens():
    enc = tiktoken.get_encoding("o200k_base")  # GPT-4o / GPT-4.1 / GPT-5 family

    # Teacher approach: full prompt + input text
    teacher_prompt_tokens = []
    for text in SAMPLE_INPUTS:
        filled = TEACHER_PROMPT.format(text=text)
        teacher_prompt_tokens.append(len(enc.encode(filled)))

    # Distilled approach: just the raw input text (+ minimal chat template overhead)
    distilled_input_tokens = []
    chat_template_overhead = 10  # <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    for text in SAMPLE_INPUTS:
        distilled_input_tokens.append(len(enc.encode(text)) + chat_template_overhead)

    # Output tokens
    teacher_output_tokens = len(enc.encode("Final Answer: en"))  # ~4 tokens
    distilled_output_tokens = len(enc.encode("en"))  # ~1 token

    avg_teacher_input = sum(teacher_prompt_tokens) / len(teacher_prompt_tokens)
    avg_distilled_input = sum(distilled_input_tokens) / len(distilled_input_tokens)

    return {
        "teacher_input_avg": avg_teacher_input,
        "teacher_input_min": min(teacher_prompt_tokens),
        "teacher_input_max": max(teacher_prompt_tokens),
        "teacher_output": teacher_output_tokens,
        "distilled_input_avg": avg_distilled_input,
        "distilled_input_min": min(distilled_input_tokens),
        "distilled_input_max": max(distilled_input_tokens),
        "distilled_output": distilled_output_tokens,
    }


# ---------------------------------------------------------------------------
# 4. Compute costs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------


def main():
    stats = count_tokens()

    print("=" * 72)
    print("LANGUAGE CLASSIFICATION: TOKEN & COST COMPARISON")
    print("=" * 72)

    # --- Token counts ---
    print("\n## Token Counts Per Request")
    print(f"{'':40s} {'Input':>10s} {'Output':>10s}")
    print(f"{'-' * 40} {'-' * 10} {'-' * 10}")
    print(
        f"{'Teacher prompt (frontier / small API)':40s} "
        f"{'~' + str(round(stats['teacher_input_avg'])):>10s} "
        f"{'~' + str(stats['teacher_output']):>10s}"
    )
    print(
        f"{'Distilled (fine-tuned, no prompt)':40s} "
        f"{'~' + str(round(stats['distilled_input_avg'])):>10s} "
        f"{'~' + str(stats['distilled_output']):>10s}"
    )
    reduction = stats["teacher_input_avg"] / stats["distilled_input_avg"]
    print(f"\n  -> Input token reduction: {reduction:.0f}x")

    # --- Per-request cost ---
    print("\n## Cost Per Request")
    header = f"{'Model':36s} {'Approach':12s} {'Input tok':>10s} {'Output tok':>10s} {'Cost/req':>12s}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, pricing in MODELS.items():
        is_tinker = "Tinker" in name
        input_tok = stats["distilled_input_avg"] if is_tinker else stats["teacher_input_avg"]
        output_tok = stats["distilled_output"] if is_tinker else stats["teacher_output"]
        approach = "distilled" if is_tinker else "teacher"
        cpr = cost_per_request(input_tok, output_tok, pricing)
        rows.append((name, approach, input_tok, output_tok, cpr))
        print(f"{name:36s} {approach:12s} {round(input_tok):>10d} {output_tok:>10d} {format_usd(cpr):>12s}")

    # --- Daily cost at volume ---
    print("\n## Daily Cost by Volume (requests/day)")
    vol_header = f"{'Model':36s}" + "".join(f" {format_k(v) + '/day':>12s}" for v in VOLUME_TIERS)
    print(vol_header)
    print("-" * len(vol_header))

    for name, approach, input_tok, output_tok, cpr in rows:
        line = f"{name:36s}"
        for vol in VOLUME_TIERS:
            daily = cpr * vol
            line += f" {format_usd(daily):>12s}"
        print(line)

    # --- Monthly cost at volume ---
    print("\n## Monthly Cost (30 days)")
    print(vol_header.replace("/day", "/day"))
    print("-" * len(vol_header))

    for name, approach, input_tok, output_tok, cpr in rows:
        line = f"{name:36s}"
        for vol in VOLUME_TIERS:
            monthly = cpr * vol * 30
            line += f" {format_usd(monthly):>12s}"
        print(line)

    # --- Savings summary ---
    frontier_cpr = rows[0][4]
    tinker_cpr = rows[-1][4]
    best_small_cpr = min(r[4] for r in rows[1:-1])
    best_small_name = [r[0] for r in rows[1:-1] if r[4] == best_small_cpr][0]

    print("\n## Savings vs Frontier (GPT-5.2)")
    print(f"  Best small API model ({best_small_name}):")
    print(f"    {frontier_cpr / best_small_cpr:.1f}x cheaper per request")
    print(f"    At 100K req/day: save {format_usd((frontier_cpr - best_small_cpr) * 100_000)}/day")
    print("\n  Fine-tuned on Tinker:")
    print(f"    {frontier_cpr / tinker_cpr:.0f}x cheaper per request")
    print(f"    At 100K req/day: save {format_usd((frontier_cpr - tinker_cpr) * 100_000)}/day")

    print()


if __name__ == "__main__":
    main()
