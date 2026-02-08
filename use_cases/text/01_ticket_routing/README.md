# Support Ticket Routing

**Type:** Single-label classification
**Labels:** `billing`, `technical`, `account`, `general`

## Task Description

Route incoming customer support tickets to the correct team. This is the **live demo** use case — universally relatable, simple 4-class classification, fast to train.

### Input/Output Examples

| Input | Output |
|---|---|
| "I was double-charged on my last invoice" | `billing` |
| "API calls returning 500 errors since the update" | `technical` |
| "Can't log into my account after password reset" | `account` |
| "Do you plan to add dark mode?" | `general` |

## How to Run

```bash
# 1. Generate training data (requires TINKER_API_KEY)
uv run use_cases/text/01_ticket_routing/create_data.py

# 2. Fine-tune the student model
uv run use_cases/text/01_ticket_routing/train.py

# 3. Evaluate (optionally pass --checkpoint for fine-tuned model)
uv run use_cases/text/01_ticket_routing/eval.py --checkpoint "tinker://<run-id>/sampler_weights/final"

# 4. View cost comparison (no API key needed)
uv run use_cases/text/01_ticket_routing/cost_comparison.py
```

## Files

- `config.py` — Task-specific config: labels, teacher prompt, token estimates, synthetic templates
- `create_data.py` — Generate training data via teacher model
- `train.py` — Fine-tune student model with LoRA
- `eval.py` — 3-model comparison evaluation
- `cost_comparison.py` — Cost analysis across model tiers
- `sample_data/` — Pre-generated examples and cost report
