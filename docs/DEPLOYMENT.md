# Deploying Fine-Tuned Models

## Serving on Tinker

The simplest path: your fine-tuned LoRA adapter runs on Tinker's infrastructure.

### Using the Checkpoint Directly

After training, your checkpoint is available at:
```
tinker://<run-id>/sampler_weights/final
```

Create a sampling client to serve it:
```python
import tinker

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(
    model_path="tinker://<run-id>/sampler_weights/final"
)
```

### Checkpoint TTL

By default, checkpoints expire after 7 days (`ttl_seconds=604800`). For production:
- Set a longer TTL during training: `train.Config(ttl_seconds=30*24*3600)` (30 days)
- Or publish weights for permanent storage (see below)

## Publishing Weights

To make weights permanent and shareable:

```bash
python -m tinker_cookbook.scripts.publish_weights \
    --checkpoint_path "tinker://<run-id>/sampler_weights/final" \
    --output_name "my_classifier_v1"
```

## Building a Simple API

Wrap your fine-tuned model in a FastAPI service:

```python
import asyncio
from fastapi import FastAPI
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

app = FastAPI()

# Initialize once at startup
service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(
    model_path="tinker://<run-id>/sampler_weights/final"
)
tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
renderer = renderers.get_renderer("qwen3", tokenizer)


@app.post("/classify")
async def classify(text: str):
    messages = [{"role": "user", "content": text}]
    model_input = renderer.build_generation_prompt(messages)
    params = tinker.SamplingParams(
        max_tokens=20,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )
    result = await sampling_client.sample_async(
        prompt=model_input, num_samples=1, sampling_params=params
    )
    response = tokenizer.decode(result.sequences[0].tokens)
    return {"label": response.strip()}
```

## Monitoring & Drift Detection

Fine-tuned models are frozen — they won't change over time. But the input distribution might:

1. **Log predictions and confidence**: Track label distribution over time
2. **Sample review**: Periodically review a random sample of predictions
3. **Retrain triggers**: If accuracy drops below threshold on spot-checks, generate new training data for failure cases and retrain
4. **A/B testing**: Run teacher model on a small percentage of traffic alongside the student to catch drift

## Cost Monitoring

Track actual vs. projected costs:
- Input tokens per request (should match `student_input_tokens` estimate)
- Output tokens per request
- Requests per day
- Compare against the cost_comparison.py projections

Run `cost_comparison.py` for any use case to see projected savings at your volume.
