# Sentiment + Aspect Extraction

**Type:** Structured JSON extraction
**Sentiment Labels:** `positive`, `negative`, `neutral`, `mixed`

## Task Description

Extract structured sentiment analysis from product reviews. Given a review, the model outputs a JSON object containing the sentiment classification, the specific aspect being discussed, a confidence score, and a brief reasoning explanation.

### Input/Output Examples

| Input | Output |
|---|---|
| "The shipping was incredibly fast, arrived in just 2 days!" | `{"sentiment": "positive", "aspect": "shipping", "confidence": 0.95, "reasoning": "Expresses strong satisfaction with delivery speed"}` |
| "Quality is terrible for the price, very disappointed." | `{"sentiment": "negative", "aspect": "quality", "confidence": 0.93, "reasoning": "Clear dissatisfaction with product quality relative to cost"}` |
| "It's an okay product, nothing special but does what it says." | `{"sentiment": "neutral", "aspect": "quality", "confidence": 0.78, "reasoning": "No strong positive or negative emotion, factual assessment"}` |
| "Love the design but the battery life is atrocious." | `{"sentiment": "mixed", "aspect": "battery", "confidence": 0.88, "reasoning": "Positive about design but strongly negative about battery life"}` |

### Output Schema

```json
{
  "sentiment": "positive | negative | neutral | mixed",
  "aspect": "string (e.g., shipping, quality, price, customer_service, packaging, durability, design, ease_of_use, battery, fit, taste, value, support, warranty)",
  "confidence": 0.0-1.0,
  "reasoning": "Brief one-sentence explanation"
}
```

## How to Run

```bash
# 1. Generate training data (requires TINKER_API_KEY)
uv run use_cases/text/04_sentiment_aspect/create_data.py

# 2. Fine-tune the student model
uv run use_cases/text/04_sentiment_aspect/train.py

# 3. Evaluate (optionally pass --checkpoint for fine-tuned model)
uv run use_cases/text/04_sentiment_aspect/eval.py --checkpoint "tinker://<run-id>/sampler_weights/final"

# 4. View cost comparison (no API key needed)
uv run use_cases/text/04_sentiment_aspect/cost_comparison.py
```

## Files

- `config.py` — Task-specific config: labels, teacher prompt, token estimates, synthetic templates
- `create_data.py` — Generate training data via teacher model
- `train.py` — Fine-tune student model with LoRA
- `eval.py` — 2-model comparison evaluation
- `cost_comparison.py` — Cost analysis across model tiers
- `cost_report.txt` — Cost report from data generation
