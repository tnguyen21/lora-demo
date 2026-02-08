---
marp: true
theme: default
paginate: true
---

# Fine-Tuning Workshop

## Small Models as Fuzzy Decision Trees

> When and how to fine-tune smaller language models for production use cases.

---

# The Problem

- **Token costs compound fast** — detailed prompts are ~500 tokens per request. At 1M req/day, that's real money.
- **Latency adds up** — frontier API calls: 500ms–2s. Fine-tuned small model: 10–50ms.
- **You're overpaying for capability** — a 70B+ param model doing binary classification is a PhD sorting mail.

**The fix:** treat small fine-tuned models as **"fuzzy decision trees."**

---

# When Fine-Tuning Makes Sense

### Good candidates

- Classification / routing
- Extraction (structured fields from unstructured text)
- Narrow chat (scoped domain)
- Format transformation (e.g., NL → SQL for a known schema)

### The pattern

1. Task is **well-scoped** with clear eval criteria
2. A large model already does it reliably (your "teacher")
3. You need **high volume** or **low latency** (or both)
4. Smaller model meets accuracy requirements (verify with evals!)

---

# When NOT to Fine-Tune

- Open-ended reasoning or creative tasks
- Requirements shift frequently (retraining has a cost)
- Low volume (just use the big model — eng time > token costs)
- You haven't validated that a large model can even do the task

---

# The Economics: Token Pricing (Feb 2026)

| Model                         | Input/1M   | Output/1M  |
| ----------------------------- | ---------- | ---------- |
| Claude Opus 4.6 (frontier)    | $5.00      | $25.00     |
| GPT-5.2 (frontier)            | $1.75      | $14.00     |
| GPT-4.1-mini                  | $0.40      | $1.60      |
| Gemini 2.5 Flash              | $0.30      | $2.50      |
| **Tinker LoRA (self-hosted)** | **~$0.10** | **~$0.10** |

---

# The Real Kicker: Prompt Compression

Raw per-token pricing is only half the story.

| Approach                    | Input tokens/req | Output tokens/req |
| --------------------------- | ---------------- | ----------------- |
| Teacher (full rubric)       | ~641             | ~4                |
| Distilled model (no prompt) | ~24              | ~1                |

**26x reduction** in input tokens per request.

Cheaper model **and** fewer tokens.

---

# Cost at Scale

| Model            | 1K req/day | 100K req/day | 1M req/day |
| ---------------- | ---------- | ------------ | ---------- |
| Claude Opus 4.6  | $3.31      | $331         | $3,305     |
| GPT-5.2          | $1.18      | $118         | $1,178     |
| Gemini 2.5 Flash | $0.20      | $20          | $202       |
| **Tinker LoRA**  | **$0.003** | **$0.25**    | **$2.52**  |

At 1M req/day: **$76/month** vs **$99K/month** (Opus) — **~1,312x savings**

---

# Break-Even Analysis

- Fine-tuning cost: **a few dollars** of compute for a LoRA adapter
- At **10K+ req/day** with a frontier model → pays for itself on day one
- At **100K+ req/day** → saving $100+/day vs frontier, $20+/day vs cheapest API model

---

# Latency: The Other Half

Transformer inference has two phases:

1. **Prefill** — process all input tokens in parallel (compute-bound)
2. **Decode** — generate output tokens one at a time (memory-bandwidth-bound)

| Phase   | Teacher     | Student                    |
| ------- | ----------- | -------------------------- |
| Prefill | ~641 tokens | ~24 tokens (27x less work) |
| Decode  | ~4 tokens   | ~1 token                   |

| Approach              | p50 Latency | Speedup  |
| --------------------- | ----------- | -------- |
| Teacher (full prompt) | ~450ms      | —        |
| Fine-tuned student    | ~35ms       | **~13x** |

---

# The Pipeline (Cheatsheet)

1. **Define your task** — inputs, outputs, eval criteria, edge cases
2. **Generate training data** — teacher model + detailed rubric → labeled JSONL
3. **Validate teacher labels** — spot-check, measure accuracy vs ground truth
4. **Fine-tune** — LoRA on a small model (`lr=1e-4`, `epochs=4`, `rank=32`)
5. **Test the student** — raw inputs, no teacher prompt
6. **Evaluate at scale** — compare accuracy, latency, cost

---

# Step 1: Define Your Task

Before any ML, write down:

- What are the **inputs**?
- What are the **outputs**?
- What does **"correct"** mean?
- What are the **edge cases**?

This becomes your **teacher prompt** and your **eval criteria**.

---

# Step 2: Generate Data with a Teacher

- Use a large model with a detailed prompt to label your data
- The teacher prompt encodes all domain knowledge (rubric, edge cases, examples)
- Save as JSONL in the **short format** — input only, no teacher prompt

> Key insight: the training data contains only the raw input → label mapping.
> The detailed rubric is **not** in the training data — it's baked into the weights.

---

# Step 3: Validate Teacher Labels

- Spot-check a sample manually
- Compute agreement with any ground truth you have
- **If the teacher can't do it reliably, the student won't either**
- Fix your prompt first, then regenerate

---

# Step 4: Fine-Tune the Student

- **LoRA** — trains a small adapter, not the full model
- Fast and cheap: minutes, not hours
- Good defaults: `lr=1e-4`, `epochs=4`, `lora_rank=32`, `batch_size=128`

---

# Step 5: Test the Fine-Tuned Model

- Load the checkpoint
- Send **raw inputs** — no detailed prompt needed
- Verify it produces the right labels

The student learned the decision boundary from examples. The teacher prompt is now in the weights.

---

# Step 6: Evaluate at Scale

Compare student vs teacher on:

- **Accuracy** — is the student within acceptable range?
- **Latency** — how much faster?
- **Cost** — input tokens reduced by how much?

---

# Prompt Distillation Pattern

The pattern is always the same:

| What      | Teacher (big model)             | Student (fine-tuned)     |
| --------- | ------------------------------- | ------------------------ |
| Prompt    | Detailed rubric + instructions  | Just the raw input       |
| Knowledge | In the prompt (500–2000 tokens) | In the weights           |
| Cost      | High (tokens × price)           | Low (few tokens × cheap) |
| Latency   | High (long prefill)             | Low (short prefill)      |

---

# Adapting to Different Tasks

| Task                        | Teacher Prompt Encodes                                   | Student Gets              |
| --------------------------- | -------------------------------------------------------- | ------------------------- |
| Ticket routing              | Routing taxonomy, team descriptions, escalation rules    | Raw customer message      |
| PII detection               | PII type definitions, regex-miss examples, context rules | Raw text                  |
| SQL generation              | Schema, SQL conventions, examples                        | Natural language question |
| Sentiment + aspect          | Aspect list, sentiment scale, edge cases                 | Review text               |
| Doc classification (vision) | Visual taxonomy, distinguishing features                 | Scanned page image        |
| Receipt extraction (vision) | Field definitions, normalization rules                   | Photo of receipt          |

---

# Use Case Catalog: Text / NLP

| Use Case                      | Why Fine-Tune?                                        |
| ----------------------------- | ----------------------------------------------------- |
| Support ticket routing        | High volume, 500+ token prompt eliminated             |
| Content moderation            | Extreme volume, inline latency constraint             |
| PII detection                 | Data can't leave infra, regex misses context          |
| Product categorization        | Deep taxonomy (100s of categories), 2K+ token prompt  |
| SQL generation (known schema) | Schema memorized in weights, no schema-in-prompt      |
| Invoice field extraction      | Company-specific schemas, huge extraction prompts     |
| Sentiment + aspect extraction | Domain-specific aspects, consistent structured output |

---

# Use Case Catalog: Vision (VLM)

| Use Case                      | Why Fine-Tune?                                     |
| ----------------------------- | -------------------------------------------------- |
| Document type classification  | OCR-free, high volume in doc-heavy industries      |
| Receipt data extraction       | Messy real-world photos, fixed known schema        |
| Fashion defect detection      | Thousands of returns/day, domain-specific taxonomy |
| Product photo quality gate    | Runs on every listing, platform-specific standards |
| Damage assessment (insurance) | Automates triage, thousands of claims/day          |
| Manufacturing QC              | Line speed, product-specific defect types          |

---

# Worked Use Cases in This Repo

### Text

1. **Support Ticket Routing** — single-label classification (live demo)
2. **PII Detection** — entity extraction (JSON)
3. **SQL Generation** — free-text generation
4. **Sentiment + Aspect** — structured JSON output

### Vision

1. **Document Type Classification** — VLM classification (live demo)
2. **Receipt Data Extraction** — VLM structured extraction

---

# Cost Summary — All 6 Use Cases

| Use Case           | Frontier $/req | Tinker $/req | Savings  |
| ------------------ | -------------- | ------------ | -------- |
| Ticket Routing     | $0.000641      | $0.000004    | **156x** |
| PII Detection      | $0.0016        | $0.000009    | **175x** |
| SQL Generation     | $0.0022        | $0.000012    | **187x** |
| Sentiment + Aspect | $0.0013        | $0.000009    | **150x** |
| Doc Classification | $0.0014        | $0.000020    | **71x**  |
| Receipt Extraction | $0.0037        | $0.000030    | **122x** |

---

# Monthly Cost at 100K req/day

| Use Case           | GPT-5.2/mo | Tinker/mo | Savings/mo |
| ------------------ | ---------- | --------- | ---------- |
| Ticket Routing     | $1,922     | $12       | $1,909     |
| PII Detection      | $4,725     | $27       | $4,698     |
| SQL Generation     | $6,720     | $36       | $6,684     |
| Sentiment + Aspect | $4,042     | $27       | $4,016     |
| Doc Classification | $4,284     | $60       | $4,224     |
| Receipt Extraction | $11,025    | $90       | $10,935    |

---

# Discussion

- **When is "good enough" good enough?**
  95% accuracy at 1/100th the cost might beat 99% for many use cases.

- **How much training data?**
  Classification: often 500–2000 examples is enough.

- **Model drift?**
  Fine-tuned model is frozen — won't degrade if the API provider updates weights.

- **Can you iterate?**
  Yes — generate more data for failure cases, retrain. The loop is fast.

---

# Key Takeaway

Fine-tuning is **prompt distillation**:

> Move knowledge from the prompt into the weights.
> Pay once to train. Save on every request forever.

**50–200x cost reduction. 10–15x latency reduction. Minutes to train.**

---

<!-- _class: lead -->

# Appendix

## ML Background for the Curious

---

# What Are "Weights"?

A neural network is a stack of matrix multiplications + nonlinearities.

```
  Input        Layer 1        Layer 2        Output
 [text] ──→ [ W₁ × x + b ] ──→ [ W₂ × x + b ] ──→ [label]
              ↑                   ↑
         these numbers       these numbers
         ARE the weights     ARE the weights
         (millions of them)
```

- **Weights** are the numbers in those matrices — millions to billions of them
- They start random. Training adjusts them so the network produces correct outputs.
- After training, the weights **encode learned knowledge** — patterns, rules, associations

Think of it like muscle memory vs. reading instructions every time.

---

# Gradient Descent: The Core Idea

**Goal:** find weight values that minimize prediction errors.

```
              ┌──────────────────────────────────────┐
              │                                      │
              ▼                                      │
  ┌───────────────────┐                              │
  │  1. Forward pass  │  "Bonjour" ──→ model ──→ "en"  (wrong!)
  └────────┬──────────┘                              │
           ▼                                         │
  ┌───────────────────┐                              │
  │  2. Compute loss  │  expected "fr", got "en" → loss = 1.0
  └────────┬──────────┘                              │
           ▼                                         │
  ┌───────────────────┐                              │
  │  3. Backward pass │  which weights caused the error?
  └────────┬──────────┘  (compute gradients)         │
           ▼                                         │
  ┌───────────────────┐                              │
  │ 4. Update weights │  nudge weights to reduce error
  └────────┬──────────┘                              │
           │                                         │
           └──── repeat for each batch ──────────────┘
```

Repeat over the full dataset = one **epoch**.

---

# Gradient Descent: Intuition

Imagine you're blindfolded on a hilly landscape, trying to reach the lowest valley.

```
  Loss
   ▲
   │\.
   │  '·.        .·'·.
   │     '·.   ·'     '·.
   │        '·'  ← you    '·..         .·'
   │         ↓ step            '·..·'
   │         ↓               (local min)
   │          ·
   │           '·.
   │              '·── goal (global min)
   └──────────────────────────────────────→ weight value
```

- Feel the slope → **gradient** | Step downhill → **weight update**
- Step size = **learning rate** (too big → overshoot, too small → slow)
- `new_weight = old_weight - learning_rate × gradient`

---

# Stochastic Gradient Descent (SGD)

Computing gradients on the _entire_ dataset per step is expensive.

```
  Full dataset (2000 examples)
  ┌──────────────────────────────────────────────────┐
  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │ ← gradient on ALL of these? slow!
  └──────────────────────────────────────────────────┘

  SGD: pick a random batch, compute gradient on just that
  ┌────────┐
  │ ▓▓▓▓▓▓ │ batch 1 → update weights
  └────────┘
       ┌────────┐
       │ ▓▓▓▓▓▓ │ batch 2 → update weights
       └────────┘
            ┌────────┐
            │ ▓▓▓▓▓▓ │ batch 3 → update weights ...
            └────────┘
```

### Key hyperparameters

- **Learning rate** — step size (1e-4 is a common starting point for fine-tuning)
- **Batch size** — examples per gradient step (larger = more stable, slower)
- **Epochs** — number of full passes through the dataset

---

# Full Fine-Tuning: The Problem

A model like Qwen3-30B has **~30 billion parameters**.

```
  Full fine-tuning: update EVERY weight in EVERY layer

  Layer 1       Layer 2       Layer 3            Layer N
  ┌─────────┐   ┌─────────┐   ┌─────────┐       ┌─────────┐
  │▓▓▓▓▓▓▓▓▓│   │▓▓▓▓▓▓▓▓▓│   │▓▓▓▓▓▓▓▓▓│  ...  │▓▓▓▓▓▓▓▓▓│
  │▓▓▓▓▓▓▓▓▓│   │▓▓▓▓▓▓▓▓▓│   │▓▓▓▓▓▓▓▓▓│       │▓▓▓▓▓▓▓▓▓│
  │▓▓▓▓▓▓▓▓▓│   │▓▓▓▓▓▓▓▓▓│   │▓▓▓▓▓▓▓▓▓│       │▓▓▓▓▓▓▓▓▓│
  └─────────┘   └─────────┘   └─────────┘       └─────────┘
   ▓ = updated    ALL 30 BILLION parameters need gradients + optimizer state

  GPU memory: model weights + gradients + optimizer = 3-4x model size
  For 30B params → need 200+ GB VRAM → multiple expensive GPUs
```

Risk **catastrophic forgetting** — overwriting general knowledge while learning your task.

For a narrow task like classification, this is overkill.

---

# LoRA: The Key Insight

**Low-Rank Adaptation (LoRA)** — Hu et al., 2021

Instead of updating the full weight matrix **W**, decompose the update into two small matrices:

```
  Full weight update ΔW          LoRA decomposition
  ┌───────────────────┐          ┌───┐
  │                   │          │   │ B
  │                   │          │   │ (d × r)
  │    d × d          │    ≈     │   │        ┌───────────────────┐
  │   (huge!)         │          │   │    ×   │     A  (r × d)    │
  │                   │          │   │        └───────────────────┘
  │                   │          └───┘
  └───────────────────┘
     16.7M params              r = 32, d = 4096 → 262K params (64x smaller)
```

- **A** has dimensions r × d (projects down to rank r)
- **B** has dimensions d × r (projects back up)
- **r** is the **LoRA rank** — typically 8, 16, or 32 (vs d = thousands)

---

# LoRA: Why It Works

```
  During inference, LoRA merges cleanly into the original weights:

                  ┌──────────────┐
                  │  W (frozen)  │─────────────────────┐
    input ────→   │  original    │                     ├──→ + ──→ output
      x           │  weights     │  ┌───┐   ┌───────┐  │
                  └──────────────┘  │ B │   │   A   │──┘
                                    │   │ × │       │
                                    └───┘   └───────┘
                                      ↑       ↑
                                  these are the ONLY
                                  things we train

  output = (W + B×A) × input
```

- **W** (frozen): 4096 × 4096 = **16.7M params** — untouched
- **A** (trained): 32 × 4096 = 131K params
- **B** (trained): 4096 × 32 = 131K params
- Trainable total: **262K** — **64x fewer** than the full matrix

---

# LoRA: The Full Picture

```
  Transformer layer (one of many)
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │   Q projection    K projection    V projection       │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐        │
  │  │ W_q ░░░░ │    │ W_k ░░░░ │    │ W_v ░░░░ │        │
  │  │ (frozen) │    │ (frozen) │    │ (frozen) │        │
  │  │ +        │    │ +        │    │ +        │        │
  │  │ B_q × A_q│    │ B_k × A_k│    │ B_v × A_v│        │
  │  │ (trained)│    │ (trained)│    │ (trained)│        │
  │  └──────────┘    └──────────┘    └──────────┘        │
  │                                                      │
  │  ░░░░ = frozen (99.8%)    trained = LoRA (0.2%)      │
  └──────────────────────────────────────────────────────┘
```

---

### Why this is great

- **Memory efficient** — only store/update the small adapter matrices
- **Fast** — fewer parameters to compute gradients for
- **Composable** — swap different LoRA adapters for different tasks on the same base model
- **No forgetting** — original weights untouched

---

# LoRA: Applied to Our Pipeline

```
  TRAINING                                INFERENCE
  ┌────────────────────────┐              ┌──────────────────────────┐
  │  Base model (30B)      │              │  Merged model            │
  │  ┌──────────────────┐  │              │  ┌──────────────────┐    │
  │  │ W ░░░░░░ FROZEN  │  │   merge      │  │ W' = W + BA      │    │
  │  │                  │  │ ────────→    │  │ (single matrix)  │    │
  │  │ + B×A  ◄─trained │  │              │  │ no overhead!     │    │
  │  └──────────────────┘  │              │  └──────────────────┘    │
  │                        │              │                          │
  │  adapter: ~50M params  │              │  Same speed as original  │
  │  (0.17% of model)      │              │  model — zero penalty    │
  │  trains in minutes     │              │                          │
  └────────────────────────┘              └──────────────────────────┘

  Hot-swap: same base model, different LoRA adapters for different tasks
  ┌──────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ Base │ + │ LoRA: tickets│ + │ LoRA: PII    │ + │ LoRA: SQL    │
  │ model│   │ (3 MB)       │   │ (3 MB)       │   │ (3 MB)       │
  └──────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

---

# Putting It All Together

```
┌─────────────────────────────────────────────────┐
│  Teacher Model (large, with detailed prompt)    │
│  "Here's a 500-token rubric for classifying..." │
│  → produces labeled examples                    │
└──────────────────────┬──────────────────────────┘
                       │ training data (JSONL)
                       ▼
┌─────────────────────────────────────────────────┐
│  LoRA Fine-Tuning                               │
│  Freeze base weights, train small A/B matrices  │
│  SGD on the labeled examples for a few epochs   │
└──────────────────────┬──────────────────────────┘
                       │ adapter weights (~few MB)
                       ▼
┌─────────────────────────────────────────────────┐
│  Student Model (small, no prompt needed)        │
│  Input: raw text → Output: label                │
│  Knowledge is in the weights, not the prompt    │
└─────────────────────────────────────────────────┘
```

---

# Appendix: Glossary

| Term              | Definition                                                                 |
| ----------------- | -------------------------------------------------------------------------- |
| **Epoch**         | One full pass through the training data                                    |
| **Batch size**    | Number of examples processed per gradient step                             |
| **Learning rate** | How big each weight update step is                                         |
| **Loss**          | A number measuring how wrong the model's predictions are                   |
| **Gradient**      | Direction and magnitude to adjust each weight to reduce loss               |
| **LoRA rank**     | Size of the low-rank decomposition (higher = more expressive, more params) |
| **Adapter**       | The small set of trained LoRA matrices (A and B)                           |
| **Prefill**       | Processing all input tokens to build the KV cache (compute-bound)          |
| **Decode**        | Generating output tokens one at a time (memory-bandwidth-bound)            |
