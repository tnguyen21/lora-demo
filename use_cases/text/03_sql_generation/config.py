"""SQL Generation — free-text generation from natural language.

Translates natural language questions into valid SQL queries against
an e-commerce database schema (users, orders, products, order_items).
"""

from shared.config import UseCaseConfig

TEACHER_PROMPT = """\
You are an expert SQL engineer. Your task is to translate a natural language question
into a single, valid SQL query against the following e-commerce database schema.

--- DATABASE SCHEMA ---

CREATE TABLE users (
    id          SERIAL PRIMARY KEY,
    email       VARCHAR(255) UNIQUE NOT NULL,
    name        VARCHAR(255) NOT NULL,
    country     VARCHAR(100),
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE products (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    category    VARCHAR(100),
    price       DECIMAL(10, 2) NOT NULL,
    stock       INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    status      VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, shipped, delivered, cancelled
    total       DECIMAL(10, 2) NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE order_items (
    id          SERIAL PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL,
    unit_price  DECIMAL(10, 2) NOT NULL
);

--- INSTRUCTIONS ---

1. Read the natural language question carefully.
2. Write a single SQL query that answers the question using the schema above.
3. Use standard SQL (PostgreSQL-compatible). Prefer explicit JOINs over implicit.
4. Use appropriate aggregate functions (COUNT, SUM, AVG, etc.) when the question
   asks for totals, averages, counts, or rankings.
5. Use DATE/TIMESTAMP functions for time-based questions. Assume the current date
   is 2025-06-15 when relative dates like "last month" appear.
6. Include ORDER BY and LIMIT when the question asks for "top N" or rankings.
7. Use clear column aliases (e.g., AS total_revenue) for computed columns.
8. Do NOT wrap the query in markdown code fences. Output ONLY the raw SQL.
9. End the query with a semicolon.

--- QUESTION ---

{input}"""

SYNTHETIC_TEMPLATES = [
    "How many orders were placed last month?",
    "What is the average order value per customer?",
    "Show the top 10 products by total revenue.",
    "List all customers who have never placed an order.",
    "What is the total revenue for each product category?",
    "Find all orders with a total greater than $500.",
    "Which customer has spent the most money overall?",
    "How many users signed up in the last 30 days?",
    "What are the top 5 most frequently ordered products?",
    "Show the monthly revenue trend for the past 12 months.",
    "List all cancelled orders along with the customer name and email.",
    "What is the average number of items per order?",
    "Find products that have never been ordered.",
    "Which country has the most registered users?",
    "Show all orders placed by users from the United States.",
    "What percentage of orders have been delivered?",
    "List the top 3 customers by number of orders placed.",
    "What is the total stock value across all products?",
    "Find all users who placed more than 5 orders.",
    "Show the revenue breakdown by order status.",
    "What is the most expensive product in the Electronics category?",
    "How many distinct products were ordered last week?",
    "List all orders that contain more than 3 different products.",
    "What is the average time between a user signing up and their first order?",
    "Show all products priced between $10 and $50 sorted by price descending.",
    "Which product category has the highest average order value?",
    "Find duplicate email addresses in the users table.",
    "What is the total number of items sold for each product?",
    "Show the top 5 countries by total order revenue.",
    "List all users who placed an order in both January and February 2025.",
    "What is the refund rate (cancelled orders / total orders)?",
    "Find the busiest day of the week by number of orders.",
    "Show all products that are currently out of stock.",
    "What is the average order total for shipped vs delivered orders?",
    "List the most recent order for each customer.",
    "How many orders were placed per day in the last 7 days?",
    "Find all customers whose name starts with 'J'.",
    "What is the total quantity sold for products in the 'Clothing' category?",
    "Show a summary of orders grouped by status with count and total revenue.",
    "Which users have ordered the product with id 42?",
]

CONFIG = UseCaseConfig(
    name="sql_generation",
    display_name="SQL Generation",
    labels=[],
    teacher_prompt=TEACHER_PROMPT,
    output_format="free_text",
    output_regex=None,
    renderer_name="qwen3",
    student_model="Qwen/Qwen3-30B-A3B",
    teacher_input_tokens=800,
    student_input_tokens=80,
    teacher_output_tokens=60,
    student_output_tokens=40,
    synthetic_examples=1000,
    eval_samples=200,
    synthetic_input_templates=SYNTHETIC_TEMPLATES,
    teacher_temperature=0.0,
)
