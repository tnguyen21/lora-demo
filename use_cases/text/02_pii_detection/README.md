# PII Detection

**Type:** Span/entity extraction with JSON output
**Labels:** `person`, `email`, `phone`, `ssn`, `address`, `credit_card`

## Task Description

Detect and extract Personally Identifiable Information (PII) from business communications such as emails, support tickets, and internal messages. The model outputs a structured JSON array of detected entities, each with the exact text span and its PII type.

### Input/Output Examples

| Input | Output |
|---|---|
| "Send the contract to John Smith at jsmith@acme.com or call 555-0123." | `[{"entity": "John Smith", "type": "person"}, {"entity": "jsmith@acme.com", "type": "email"}, {"entity": "555-0123", "type": "phone"}]` |
| "Process refund for SSN 456-78-9012, card 4532-8810-1234-5678." | `[{"entity": "456-78-9012", "type": "ssn"}, {"entity": "4532-8810-1234-5678", "type": "credit_card"}]` |
| "The quarterly report is ready for review." | `[]` |

## How to Run

```bash
# 1. Generate training data (requires TINKER_API_KEY)
python use_cases/text/02_pii_detection/create_data.py

# 2. Fine-tune the student model
python use_cases/text/02_pii_detection/train.py

# 3. Evaluate (optionally pass --checkpoint for fine-tuned model)
python use_cases/text/02_pii_detection/eval.py --checkpoint "tinker://<run-id>/sampler_weights/final"

# 4. View cost comparison (no API key needed)
python use_cases/text/02_pii_detection/cost_comparison.py
```

## Files

- `config.py` — Task-specific config: PII labels, teacher prompt with detection rubric, token estimates, synthetic templates
- `create_data.py` — Generate training data via teacher model
- `train.py` — Fine-tune student model with LoRA
- `eval.py` — 3-model comparison evaluation
- `cost_comparison.py` — Cost analysis across model tiers
- `sample_data/` — Pre-generated examples for reference
