# Fine-Tuning Workshop: Small Models as Fuzzy Decision Trees

> Workshop for engineers and founders on when and how to fine-tune smaller language models for production use cases.

---

## Core Thesis

Large language models (GPT-4o, Claude Sonnet, Gemini Pro) can one-shot most narrow tasks — classification, extraction, routing, formatting. But at scale, the math doesn't work:

- **Token costs compound fast.** A detailed classification prompt is ~500 tokens input per request. At 1M requests/day, that's real money.
- **Latency adds up.** Large model API calls are 500ms-2s. A fine-tuned small model can be 10-50ms.
- **You're paying for capabilities you don't need.** A 70B+ parameter model doing binary classification is like hiring a PhD to sort mail.

The alternative: **treat small fine-tuned models as "fuzzy decision trees."** They learn the decision boundary from examples, handle edge cases better than rule-based systems, and run at a fraction of the cost.

---

## When Fine-Tuning Makes Sense

### Good candidates (the sweet spot)

- **Classification / routing** — language detection, intent classification, content moderation, ticket routing
- **Extraction** — pulling structured fields from unstructured text (dates, entities, amounts)
- **Narrow chat** — customer service agents scoped to a specific domain (returns policy, billing FAQ)
- **Format transformation** — converting between formats where the mapping is learnable (e.g., natural language to SQL for a known schema)

### Common pattern

1. The task is **well-scoped** — you can write clear evaluation criteria
2. A large model can already do it reliably (this is your "teacher")
3. You need it at **high volume** or **low latency** (or both)
4. Accuracy requirements are met by the smaller model (verify with evals!)

### When NOT to fine-tune

- Open-ended reasoning or creative tasks
- Tasks where requirements shift frequently (retraining has a cost)
- Low volume (just use the big model — engineering time > token costs)
- When you haven't validated that a large model can even do the task well

---

## The Economics: Why This Matters

### Raw token pricing (per 1M tokens, Feb 2026)

| Provider / Model                         | Input  | Output |
| ---------------------------------------- | ------ | ------ |
| OpenAI GPT-5.2 (frontier)                | $1.75  | $14.00 |
| OpenAI GPT-5 (frontier)                  | $1.25  | $10.00 |
| Anthropic Claude Opus 4.6 (frontier)     | $5.00  | $25.00 |
| OpenAI GPT-4.1-mini                      | $0.40  | $1.60  |
| Google Gemini 2.5 Flash                  | $0.30  | $2.50  |
| Anthropic Claude Haiku 4.5               | $1.00  | $5.00  |
| Tinker Qwen3-30B-A3B (LoRA, self-hosted) | ~$0.10 | ~$0.10 |

_Tinker self-hosted pricing is estimated from GPU serving costs and varies by utilization. Run `scripts/cost_comparison.py` to compute with your actual rates._

### The real kicker: prompt compression

Raw per-token pricing only tells half the story. With a large model, you need a detailed prompt (500-2000 tokens) to get reliable behavior. A fine-tuned model has that knowledge **baked into the weights** — your prompt can be just the input data. This is **prompt distillation** in practice.

**Measured token counts from this workshop's demo** (via `scripts/cost_comparison.py`):

| Approach                                    | Input tokens/req | Output tokens/req |
| ------------------------------------------- | ---------------- | ----------------- |
| Teacher prompt (full classification rubric) | ~641             | ~4                |
| Distilled model (fine-tuned, no prompt)     | ~24              | ~1                |

That's a **26x reduction** in input tokens per request. The savings compound: cheaper model _and_ fewer tokens.

### What this actually costs at scale

| Model                                  | 1K req/day | 10K req/day | 100K req/day | 1M req/day |
| -------------------------------------- | ---------- | ----------- | ------------ | ---------- |
| GPT-5.2 (frontier + teacher prompt)    | $1.18      | $11.78      | $118         | $1,178     |
| GPT-5 (frontier + teacher prompt)      | $0.84      | $8.41       | $84          | $841       |
| Claude Opus 4.6 (+ teacher prompt)     | $3.31      | $33.05      | $331         | $3,305     |
| GPT-4.1-mini (+ teacher prompt)        | $0.26      | $2.63       | $26          | $263       |
| Gemini 2.5 Flash (+ teacher prompt)    | $0.20      | $2.02       | $20          | $202       |
| Claude Haiku 4.5 (+ teacher prompt)    | $0.66      | $6.61       | $66          | $661       |
| **Tinker LoRA (distilled, no prompt)** | **$0.003** | **$0.03**   | **$0.25**    | **$2.52**  |

At 1M requests/day, the fine-tuned model costs **$76/month** vs **$99K/month** for the most expensive frontier model (Opus 4.6) — a **~1,312x cost reduction**.

Even compared to the cheapest small API model (Gemini 2.5 Flash at $6K/month), the fine-tuned model is **~80x cheaper** because you're saving on both the per-token rate and the token count.

### Break-even analysis

Fine-tuning cost: a few dollars of compute for a LoRA adapter on Tinker.

If you're making 10K+ requests/day with a frontier model, fine-tuning pays for itself on day one. At 100K+ requests/day, you're saving $100+/day compared to the frontier model — and still $20+/day compared to the cheapest small API model.

> Run `scripts/cost_comparison.py` to regenerate these numbers with updated pricing.

### Latency: the other half of the story

Cost is only half the win. The fine-tuned model is also dramatically faster because of how transformer inference works.

**Transformer inference has two phases:**

1. **Prefill** — process all input tokens in parallel to build the KV cache. This is **compute-bound**: FLOPs scale with sequence length (quadratically for attention, linearly for FFN layers). More input tokens = more arithmetic.
2. **Decode** — generate output tokens one at a time, autoregressively. This is **memory-bandwidth-bound**: each step loads the full model weights from GPU memory to produce a single token. Decode cost is roughly constant regardless of how long the input was.

**Where the speedup comes from:** the teacher prompt is ~641 tokens (full rubric + input). The fine-tuned student prompt is ~24 tokens (just the raw input). That's a ~27x reduction in prefill work. Decode is nearly identical for both — same model architecture, same short output. The entire latency gap comes from prefill savings.

| Phase | Teacher | Student | Bottleneck |
| ------- | ------------------- | ---------------- | -------------- |
| Prefill | ~641 tokens | ~24 tokens | Compute-bound (27x less work) |
| Decode | ~4 output tokens | ~1 output token | Memory-bandwidth-bound (similar cost) |

| Approach                              | p50 Latency | Speedup |
| ------------------------------------- | ----------- | ------- |
| Teacher (Qwen3-30B-A3B + full prompt) | ~450ms      | —       |
| Fine-tuned student (no prompt)        | ~35ms       | ~13x    |

> Run `scripts/benchmark_latency.py` to measure on your hardware.

---

## Cheatsheet

Quick reference for the fine-tuning pipeline. Each step links to the detailed walkthrough below.

- **[Setup](#step-0-setup)** — `pip install tinker-cookbook` and set your API key.
- **[Define your task](#step-1-define-your-task--write-a-rubric)** — Write down inputs, outputs, eval criteria, and edge cases before touching any code.
- **[Generate training data](#step-2-generate-training-data-with-a-teacher-model)** — Prompt a large model with a detailed rubric to label your dataset. Save as JSONL in the short format (no teacher prompt in the training data).
- **[Validate teacher labels](#step-3-run-evals-on-teacher-output-sanity-check)** — Spot-check and measure accuracy against any ground truth. If the teacher is wrong, fix the prompt — don't train on bad labels.
- **[Fine-tune](#step-4-fine-tune-the-student)** — LoRA on a small model. `learning_rate=1e-4`, `num_epochs=4`, `lora_rank=32` are good starting points.
- **[Test the student](#step-5-test-the-fine-tuned-model)** — Load the checkpoint and send raw inputs (no teacher prompt). Verify it produces the right labels.
- **[Evaluate at scale](#step-6-evaluate-at-scale)** — Compare student vs teacher on accuracy, latency, and cost. The student should be close on accuracy and dramatically cheaper.

---

## Workshop Walkthrough

We'll build an end-to-end pipeline using a real task: **multilingual language classification.**

### Step 0: Setup

```bash
pip install tinker-cookbook
export TINKER_API_KEY="your-key-here"
```

### Step 1: Define Your Task + Write a Rubric

Before any ML, write down:

- What are the inputs? (text strings)
- What are the outputs? (one of 13 language codes)
- What does "correct" mean? (exact match on language code)
- What are the edge cases? (mixed-language text, transliterated text, code snippets)

This rubric becomes your **teacher prompt** and your **eval criteria.**

### Step 2: Generate Training Data with a Teacher Model

Use a large model with a detailed prompt to label your data:

```python
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# The teacher: a capable model with a detailed prompt
service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-30B-A3B")
tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
renderer = renderers.get_renderer("qwen3", tokenizer)

TEACHER_PROMPT = """You are a precise language classifier.
Classify the language of the provided text into exactly one of these labels:
ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot.
... (detailed rubric with edge cases) ...

Text: {text}
"""

# Generate labels
params = tinker.SamplingParams(max_tokens=1000, temperature=0.15,
                                stop=renderer.get_stop_sequences())
model_input = tinker.ModelInput.from_ints(tokenizer.encode(TEACHER_PROMPT.format(text=sentence)))
result = sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=params).result()
label = tokenizer.decode(result.sequences[0].tokens)
```

Save as JSONL with the **short format** (input only, no teacher prompt):

```json
{
  "messages": [
    { "role": "user", "content": "Bonjour, comment allez-vous?" },
    { "role": "assistant", "content": "fr" }
  ]
}
```

> See full script: `tinker_cookbook/recipes/prompt_distillation/create_data.py`

### Step 3: Run Evals on Teacher Output (Sanity Check)

Before training, verify the teacher's labels are good. Spot-check a sample, compute agreement with any ground truth you have.

```python
# Simple accuracy check against known labels
correct = sum(1 for pred, gold in zip(predictions, ground_truth) if pred == gold)
print(f"Teacher accuracy: {correct}/{len(ground_truth)}")
```

If the teacher can't do it reliably, the student won't either. Fix your prompt first.

### Step 4: Fine-Tune the Student

```bash
python -m tinker_cookbook.recipes.prompt_distillation.train \
    --file_path /tmp/tinker-datasets/prompt_distillation_lang.jsonl \
    --model_name Qwen/Qwen3-30B-A3B \
    --learning_rate 1e-4 \
    --num_epochs 4 \
    --lora_rank 32 \
    --batch_size 128
```

Key points:

- **LoRA** keeps fine-tuning fast and cheap (only trains a small adapter, not the full model)
- Learning rate of 1e-4 is a good starting point for LoRA
- 4 epochs is usually enough for classification tasks
- Training takes minutes, not hours

### Step 5: Test the Fine-Tuned Model

```python
# Load the fine-tuned checkpoint
sampling_client = service_client.create_sampling_client(
    model_path="tinker://<your-run-id>/sampler_weights/final"
)

# Now just send the raw input — no detailed prompt needed!
messages = [{"role": "user", "content": "Bonjour, comment allez-vous?"}]
model_input = renderer.build_generation_prompt(messages)

params = tinker.SamplingParams(max_tokens=20, temperature=0.15,
                                stop=renderer.get_stop_sequences())
result = sampling_client.sample(prompt=model_input, num_samples=1,
                                sampling_params=params).result()
response = tokenizer.decode(result.sequences[0].tokens)
# -> "fr"
```

> See full eval script: `scripts/sample_test_model.py`

### Step 6: Evaluate at Scale

Run your eval suite on both the teacher (with full prompt) and the student (without prompt). Compare:

- **Accuracy** — is the student within acceptable range?
- **Latency** — how much faster?
- **Cost** — input tokens reduced by how much?

---

## Template: Adapt This to Your Use Case

The pattern is always the same:

1. **Define the task** — inputs, outputs, eval criteria
2. **Write a teacher prompt** — detailed instructions that make a large model reliable
3. **Generate labeled data** — use the teacher to label your dataset
4. **Validate labels** — spot-check, compute metrics against any ground truth
5. **Fine-tune** — LoRA on a small model using the labeled data
6. **Evaluate** — compare student vs teacher on held-out data
7. **Deploy** — serve the fine-tuned model, monitor for drift

### Adapting for different tasks

The teacher prompt encodes all the domain knowledge; the student learns to skip it. Here are concrete examples of how the prompt-distillation pattern maps to different tasks:

| Task                                  | Teacher Prompt (what the big model needs)                                       | Student Input                                                  | Student Output                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------- |
| Support ticket routing                | Routing taxonomy, team descriptions, escalation rules, example tickets          | `"I was double-charged on my last invoice"`                    | `billing`                                                         |
| Content moderation                    | Policy document, edge-case examples, severity definitions                       | `"I'm going to find you and make you pay"`                     | `unsafe:threat`                                                   |
| PII detection                         | PII type definitions, regex-miss examples, context rules for ambiguous patterns | `"Call me at 555-0123 or email jd@acme.com"`                   | `{"phone": "555-0123", "email": "jd@acme.com"}`                   |
| Product categorization                | Full taxonomy tree, mapping rules, examples per category                        | `"Organic cold-pressed green juice 16oz"`                      | `grocery > beverages > juice`                                     |
| Invoice field extraction              | Field definitions, format normalization rules, multi-line item handling         | `"Invoice #4821 from Acme Corp, 2025-12-01, Total: $3,420.00"` | `{"vendor": "Acme Corp", "date": "2025-12-01", "total": 3420.00}` |
| Document type classification (vision) | Visual taxonomy, examples of each doc type, distinguishing features             | _(scanned page image)_                                         | `w2_form`                                                         |

---

## Use Case Catalog

Concrete use cases where fine-tuning a small model beats prompting a large one. Use this to pattern-match to your own problem.

### Text / NLP

| #   | Use Case                              | Who Cares                                         | Input → Output                                                                                | Why Fine-Tune?                                                                                                         |
| --- | ------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 1   | **Support ticket routing**            | Every SaaS with a support queue                   | Customer message → team label (`billing`, `technical`, `account`, `security`)                 | High volume (thousands/day). Detailed routing taxonomy is 500+ tokens of prompt you can eliminate.                     |
| 2   | **Content moderation**                | Marketplaces, social platforms, UGC apps          | User-submitted text → `safe` / `unsafe` / `needs_review` + category                           | Extreme volume, latency-sensitive (inline with every post). Can't afford 500ms per check.                              |
| 3   | **Lead qualification**                | Sales-led B2B startups                            | Form submission or chat transcript → score (`hot`, `warm`, `cold`) + reason                   | Moderate volume but high business value per classification. Scoring rubric is complex and proprietary.                 |
| 4   | **Email intent classification**       | CRM platforms, helpdesks                          | Email body → intent (`complaint`, `question`, `cancellation`, `upgrade_request`, `spam`)      | Feeds downstream automation. Needs to run on every inbound email without adding latency to the pipeline.               |
| 5   | **PII detection & redaction**         | Healthcare, fintech, any compliance-sensitive app | Raw text → tagged spans (`SSN`, `email`, `phone`, `address`)                                  | Data can't leave your infra. Must be fast (inline with every write). Regex misses context-dependent PII.               |
| 6   | **Product categorization**            | Marketplaces, e-commerce aggregators              | Product title + description → taxonomy path (`electronics > phones > accessories`)            | Thousands of new listings/day. Taxonomy is deep (100s of categories). Teacher prompt with full taxonomy is 2K+ tokens. |
| 7   | **Sentiment + aspect extraction**     | Brand monitoring, review aggregation              | Review text → sentiment per aspect (`{"shipping": "negative", "quality": "positive"}`)        | Aspect list is domain-specific. Volume scales with review count. Structured output must be consistent.                 |
| 8   | **Chat escalation detection**         | Contact centers, chatbot platforms                | Bot conversation history → `escalate` / `continue` + confidence                               | Latency-critical: runs inline with every bot message. Must react in <50ms.                                             |
| 9   | **Invoice field extraction**          | Expense management, AP automation                 | OCR text from invoice → structured JSON (`vendor`, `amount`, `date`, `line_items`)            | Field schemas are company-specific. Thousands of invoices/month. Teacher prompt with extraction rules is huge.         |
| 10  | **SQL generation for a known schema** | Internal analytics tools, BI platforms            | Natural language question → SQL query for a specific database                                 | Schema is fixed — the model can memorize table/column names. Eliminates schema-in-prompt overhead.                     |
| 11  | **Log anomaly classification**        | DevOps, observability platforms                   | Log line → severity (`normal`, `warn`, `error`, `critical`) + category                        | Extreme volume (millions of lines/day). Must be cheap per classification. Log formats are domain-specific.             |
| 12  | **Webhook / event routing**           | Integration platforms, event-driven architectures | Incoming JSON payload → handler label (`payment_received`, `user_signup`, `inventory_update`) | High volume, low latency. Payload shapes are known. Routing logic is learnable from examples.                          |

### Vision (VLM)

| #   | Use Case                               | Who Cares                                     | Input → Output                                                                                     | Why Fine-Tune?                                                                                   |
| --- | -------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | **Fashion defect detection**           | Returns processing, resale platforms          | Photo of garment → defect labels (`stain`, `tear`, `missing_button`, `none`)                       | Real-world example: processing thousands of returns/day. Defect taxonomy is domain-specific.     |
| 2   | **Product photo quality gate**         | Marketplaces, e-commerce platforms            | Seller-uploaded photo → `pass` / `reject` + reasons (`bad_lighting`, `watermark`, `wrong_angle`)   | Runs on every new listing. Standards are platform-specific. Saves manual review hours.           |
| 3   | **Damage assessment (insurance)**      | Insurance companies, claims platforms         | Claim photo (vehicle, property) → severity (`minor`, `moderate`, `severe`, `totaled`)              | Automates initial triage. Thousands of claims/day during peak events. Reduces adjuster workload. |
| 4   | **Document type classification**       | Document processing pipelines, lending, legal | Scanned page image → doc type (`invoice`, `contract`, `w2`, `drivers_license`, `bank_statement`)   | OCR-free classification from the image itself. High volume in document-heavy industries.         |
| 5   | **Receipt / invoice data extraction**  | Expense apps, accounting automation           | Photo of receipt → structured JSON (`merchant`, `total`, `date`, `items`)                          | Handles messy real-world photos (crumpled, angled, faded). Schema is fixed and known.            |
| 6   | **Shelf compliance / planogram audit** | Retail operations, CPG brands                 | Photo of store shelf → compliance report (`product_x: correct_position`, `product_y: missing`)     | Replaces manual store audits. Planogram is store-specific — perfect for fine-tuning.             |
| 7   | **Manufacturing QC**                   | Factory automation, industrial IoT            | Assembly line photo → `pass` / `fail` + defect type (`misaligned`, `scratch`, `missing_component`) | Must run at line speed. Defect types are product-specific. Volume is continuous.                 |
| 8   | **Safety / PPE compliance**            | Construction, warehouses, mining              | Security camera frame → PPE status (`hard_hat: missing`, `vest: present`, `goggles: missing`)      | Runs on live video frames. Site-specific requirements. Must be fast and cheap per frame.         |

> **Workshop tip:** Pick the use case closest to your problem, swap in your labels and data, and follow the same 7-step pattern from the template above.

---

## Discussion Points for the Workshop

- **When is "good enough" good enough?** A 95% accurate model at 1/100th the cost might beat a 99% model for many use cases.
- **How much training data do you need?** For classification, often 500-2000 examples is enough. For generation tasks, more.
- **What about model drift?** Your fine-tuned model is frozen — it won't degrade over time like a prompted model might if the API provider updates weights.
- **Can you iterate?** Yes — generate more data for failure cases, retrain. The loop is fast.

---

## Worked Use Cases

The repo includes 6 fully-worked use cases, each with data generation, training, evaluation, and cost comparison scripts:

### Text

| #   | Use Case                               | Type                        | Directory                                                                    |
| --- | -------------------------------------- | --------------------------- | ---------------------------------------------------------------------------- |
| 1   | **Support Ticket Routing** (live demo) | Single-label classification | [`use_cases/text/01_ticket_routing/`](use_cases/text/01_ticket_routing/)     |
| 2   | **PII Detection**                      | Entity extraction (JSON)    | [`use_cases/text/02_pii_detection/`](use_cases/text/02_pii_detection/)       |
| 3   | **SQL Generation**                     | Free-text generation        | [`use_cases/text/03_sql_generation/`](use_cases/text/03_sql_generation/)     |
| 4   | **Sentiment + Aspect Extraction**      | Structured JSON output      | [`use_cases/text/04_sentiment_aspect/`](use_cases/text/04_sentiment_aspect/) |

### Vision

| #   | Use Case                                     | Type                      | Directory                                                                            |
| --- | -------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| 1   | **Document Type Classification** (live demo) | VLM classification        | [`use_cases/vision/01_document_type/`](use_cases/vision/01_document_type/)           |
| 2   | **Receipt Data Extraction**                  | VLM structured extraction | [`use_cases/vision/02_receipt_extraction/`](use_cases/vision/02_receipt_extraction/) |

### Additional Resources

- [Failure Modes & Debugging](docs/FAILURE_MODES.md) — common training failures + fixes
- [Deployment Guide](docs/DEPLOYMENT.md) — how to serve fine-tuned models in production
- Run `python scripts/generate_summary_table.py` to see cost comparisons across all 6 use cases

# Cost Comparison Summary — All Use Cases

| Use Case                      | Category | Teacher In/Out | Student In/Out | Frontier $/req | Tinker $/req | Savings |
| ----------------------------- | -------- | -------------- | -------------- | -------------- | ------------ | ------- |
| Support Ticket Routing        | text     | 350/2          | 40/1           | $0.000641      | $0.000004    | 156x    |
| PII Detection                 | text     | 500/50         | 60/30          | $0.0016        | $0.000009    | 175x    |
| SQL Generation                | text     | 800/60         | 80/40          | $0.0022        | $0.000012    | 187x    |
| Sentiment + Aspect Extraction | text     | 450/40         | 60/30          | $0.0013        | $0.000009    | 150x    |
| Document Type Classification  | vision   | 800/2          | 200/1          | $0.0014        | $0.000020    | 71x     |
| Receipt Data Extraction       | vision   | 900/150        | 200/100        | $0.0037        | $0.000030    | 122x    |

## Monthly Cost at 100K requests/day

| Use Case                      | GPT-5.2/mo | Tinker/mo | Monthly Savings |
| ----------------------------- | ---------- | --------- | --------------- |
| Support Ticket Routing        | $1,922     | $12.30    | $1,909          |
| PII Detection                 | $4,725     | $27.00    | $4,698          |
| SQL Generation                | $6,720     | $36.00    | $6,684          |
| Sentiment + Aspect Extraction | $4,042     | $27.00    | $4,016          |
| Document Type Classification  | $4,284     | $60.30    | $4,224          |
| Receipt Data Extraction       | $11,025    | $90.00    | $10,935         |
