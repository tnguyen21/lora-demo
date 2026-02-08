# Receipt Data Extraction

**Type:** VLM structured field extraction (vision)
**Output:** JSON with `vendor`, `date`, `total`, `line_items`

## Task Description

Extract structured data from receipt images. The model outputs a JSON object with vendor name, date, total amount, and itemized line items.

### Input/Output Examples

| Input | Output |
|---|---|
| *(Walmart receipt image)* | `{"vendor": "Walmart", "date": "2025-12-15", "total": 11.34, "line_items": [...]}` |
| *(Starbucks receipt image)* | `{"vendor": "Starbucks", "date": "2025-12-14", "total": 9.69, "line_items": [...]}` |

## How to Run

```bash
# 1. Generate training data (requires receipt images + TINKER_API_KEY)
uv run use_cases/vision/02_receipt_extraction/create_data.py

# 2. Fine-tune the VLM student model
uv run use_cases/vision/02_receipt_extraction/train.py

# 3. Evaluate
uv run use_cases/vision/02_receipt_extraction/eval.py --checkpoint "tinker://<run-id>/sampler_weights/final"

# 4. View cost comparison (no API key needed)
uv run use_cases/vision/02_receipt_extraction/cost_comparison.py
```

## Files

- `config.py` — Task-specific config: extraction fields, VLM teacher prompt, token estimates
- `create_data.py` — Generate training data via VLM teacher model
- `train.py` — Fine-tune VLM student model with LoRA
- `eval.py` — 2-model comparison evaluation
- `cost_comparison.py` — Cost analysis across model tiers
- `sample_data/` — Pre-generated examples and cost report
