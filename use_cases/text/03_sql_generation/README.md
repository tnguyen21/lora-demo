# SQL Generation

**Type:** Free-text generation
**Labels:** None (open-ended SQL output)

## Task Description

Translate natural language questions into valid SQL queries against an e-commerce database schema. The schema contains four tables: `users`, `orders`, `products`, and `order_items`.

The teacher prompt includes the full database schema so the model learns the table structures, column names, and relationships. The student model learns to generate correct SQL from just the natural language question.

### Input/Output Examples

| Input | Output |
|---|---|
| "How many orders were placed last month?" | `SELECT COUNT(*) AS order_count FROM orders WHERE created_at >= '2025-05-01' AND created_at < '2025-06-01';` |
| "Show the top 10 products by total revenue." | `SELECT p.name, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM products p JOIN order_items oi ON oi.product_id = p.id GROUP BY p.id, p.name ORDER BY total_revenue DESC LIMIT 10;` |
| "List all customers who have never placed an order." | `SELECT u.id, u.name, u.email FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE o.id IS NULL;` |
| "What percentage of orders have been delivered?" | `SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'delivered') / COUNT(*), 2) AS delivered_pct FROM orders;` |

### Database Schema

```sql
users       (id, email, name, country, created_at)
products    (id, name, category, price, stock, created_at)
orders      (id, user_id, status, total, created_at)
order_items (id, order_id, product_id, quantity, unit_price)
```

## How to Run

```bash
# 1. Generate training data (requires TINKER_API_KEY)
uv run use_cases/text/03_sql_generation/create_data.py

# 2. Fine-tune the student model
uv run use_cases/text/03_sql_generation/train.py

# 3. Evaluate (optionally pass --checkpoint for fine-tuned model)
uv run use_cases/text/03_sql_generation/eval.py --checkpoint "tinker://<run-id>/sampler_weights/final"

# 4. View cost comparison (no API key needed)
uv run use_cases/text/03_sql_generation/cost_comparison.py
```

## Files

- `config.py` -- Task-specific config: teacher prompt with DB schema, token estimates, synthetic templates
- `create_data.py` -- Generate training data via teacher model
- `train.py` -- Fine-tune student model with LoRA
- `eval.py` -- 3-model comparison evaluation
- `cost_comparison.py` -- Cost analysis across model tiers
- `sample_data/` -- Pre-generated examples
