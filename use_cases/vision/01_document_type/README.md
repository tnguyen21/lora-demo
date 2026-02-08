# Document Type Classification

**Type:** VLM single-label classification (vision)
**Labels:** `invoice`, `receipt`, `letter`, `form`, `resume`

## Task Description

Classify scanned document images into document types. This is the **vision live demo** use case — good public datasets (RVL-CDIP), proven VLM classifier path, easy to evaluate.

### Input/Output Examples

| Input | Output |
|---|---|
| *(scanned invoice with line items and total)* | `invoice` |
| *(grocery store receipt with item list)* | `receipt` |
| *(formal business letter on letterhead)* | `letter` |
| *(job application form with blank fields)* | `form` |
| *(professional resume with experience sections)* | `resume` |

## How to Run

```bash
# 1. Generate training data (requires images + TINKER_API_KEY)
uv run use_cases/vision/01_document_type/create_data.py

# 2. Fine-tune the VLM student model
uv run use_cases/vision/01_document_type/train.py

# 3. Evaluate
uv run use_cases/vision/01_document_type/eval.py --checkpoint "tinker://<run-id>/sampler_weights/final"

# 4. View cost comparison (no API key needed)
uv run use_cases/vision/01_document_type/cost_comparison.py
```

## Files

- `config.py` — Task-specific config: labels, VLM teacher prompt, token estimates
- `create_data.py` — Generate training data via VLM teacher model
- `train.py` — Fine-tune VLM student model with LoRA
- `eval.py` — 2-model comparison evaluation
- `cost_comparison.py` — Cost analysis across model tiers
- `sample_data/` — Pre-generated examples and cost report
