"""Receipt Data Extraction — VLM structured field extraction.

Extracts vendor, date, total, and line_items from receipt images.
Output format: JSON with structured fields.
"""

from shared.config import VisionUseCaseConfig

TEACHER_PROMPT = """\
You are an expert receipt data extractor. Given an image of a receipt, extract the following structured information:

Fields to extract:
- vendor: The name of the store/business (string)
- date: The transaction date in YYYY-MM-DD format (string, or null if not visible)
- total: The total amount paid as a number (float, or null if not visible)
- line_items: A list of items purchased, each with:
  - name: Item description (string)
  - price: Item price as a number (float)

Instructions:
1. Extract the vendor name from the top of the receipt (store name, not address).
2. Find the transaction date. Convert to YYYY-MM-DD format. If only partial date visible, do your best.
3. The total should be the final amount (after tax, discounts). Look for "Total", "Amount Due", "Grand Total".
4. For line items, extract each distinct item with its price. Skip subtotals, tax lines, and payment method lines.
5. If a field is not visible or unreadable, use null.
6. Prices should be numbers without currency symbols (e.g., 4.99 not "$4.99").

Output format:
Respond with a JSON object exactly matching this structure:
{{"vendor": "...", "date": "...", "total": ..., "line_items": [{{"name": "...", "price": ...}}, ...]}}

{input}"""

SYNTHETIC_TEMPLATES = [
    "A Walmart receipt showing groceries: milk $3.99, bread $2.49, eggs $4.29, total $11.34",
    "A Starbucks receipt for a grande latte $5.75 and blueberry muffin $3.45, total $9.69",
    "A Target receipt with household items: detergent $12.99, paper towels $8.49, total $22.74",
    "A CVS pharmacy receipt for prescriptions and over-the-counter medication totaling $45.67",
    "A restaurant receipt from Olive Garden: two entrees, appetizer, and drinks, total $67.82",
    "A gas station receipt showing 12.5 gallons of regular unleaded at $3.89/gal, total $48.63",
    "A Home Depot receipt for lumber, screws, and paint supplies, total $156.23",
    "A McDonald's drive-thru receipt: 2 Big Macs, fries, drinks, total $18.45",
    "An Amazon Fresh delivery receipt with 15 grocery items, total $89.34",
    "A Costco receipt showing bulk items: toilet paper, granola bars, chicken, total $134.56",
    "A dry cleaning receipt: 3 suits, 5 shirts, 2 dresses, total $78.50",
    "A FedEx shipping receipt for 2 packages, overnight delivery, total $45.90",
    "A hotel minibar receipt: water, snacks, wine, total $32.00",
    "A pet store receipt: dog food, treats, toy, total $42.15",
    "An auto parts store receipt: oil filter, brake pads, wiper blades, total $67.89",
    "A bookstore receipt: 3 paperbacks and a magazine, total $38.97",
    "A Subway receipt: footlong sandwich, cookie, drink, total $14.23",
    "A hardware store receipt for plumbing supplies, total $89.45",
    "A clothing store receipt: jeans $49.99, t-shirt $19.99, socks $8.99, total $83.85",
    "A electronics store receipt: USB cable $12.99, phone case $24.99, total $40.08",
    "A Whole Foods receipt with organic produce and deli items, total $67.34",
    "A Uber Eats receipt showing delivery fee, service fee, and food total $34.56",
    "A sporting goods store receipt: running shoes $129.99, socks $12.99, total $151.40",
    "A bakery receipt: birthday cake $35.00, cupcakes $18.00, total $56.13",
    "A movie theater receipt: 2 tickets, popcorn, drinks, total $42.50",
    "A car wash receipt: premium wash $24.99, air freshener $3.99, total $30.67",
    "A office supply store receipt: printer paper, pens, folders, total $45.23",
    "A flower shop receipt: bouquet $45.00, vase $15.00, delivery $10.00, total $74.38",
    "A tire shop receipt: 4 tires installed and balanced, total $589.96",
    "A grocery store receipt from Trader Joe's with snacks and frozen meals, total $52.18",
    "A salon receipt: haircut $35.00, color $85.00, tip $24.00, total $152.54",
    "A parking garage receipt: 4 hours at downtown lot, total $28.00",
    "A craft store receipt: yarn, needles, pattern book, total $34.67",
    "A pizza delivery receipt: large pepperoni, garlic knots, total $28.45",
    "A veterinary clinic receipt: exam, vaccination, flea treatment, total $187.50",
    "A liquor store receipt: wine, beer, spirits, total $65.43",
    "A toy store receipt: board game, action figure, puzzle, total $52.97",
    "A shoe repair receipt: resole and heel repair, total $45.00",
    "A phone repair shop receipt: screen replacement for iPhone, total $149.99",
    "A laundromat receipt: wash and fold service 15 lbs, total $22.50",
]

CONFIG = VisionUseCaseConfig(
    name="receipt_extraction",
    display_name="Receipt Data Extraction",
    labels=["vendor", "date", "total", "line_items"],
    teacher_prompt=TEACHER_PROMPT,
    output_format="json",
    teacher_input_tokens=900,
    student_input_tokens=200,
    teacher_output_tokens=150,
    student_output_tokens=100,
    synthetic_examples=500,
    eval_samples=100,
    synthetic_input_templates=SYNTHETIC_TEMPLATES,
    data_source="sroie",
    teacher_max_tokens=2000,
)
