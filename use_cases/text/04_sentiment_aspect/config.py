"""Sentiment + Aspect Extraction — structured JSON output.

4 sentiment labels: positive, negative, neutral, mixed
Extracts: sentiment, aspect, confidence, reasoning from product reviews.
"""

from shared.config import UseCaseConfig

TEACHER_PROMPT = """\
You are an expert product review analyst specializing in sentiment and aspect extraction.

Goal: Analyze the product review and extract structured sentiment information as a JSON object.

Output JSON schema:
- sentiment: exactly one of "positive", "negative", "neutral", or "mixed"
- aspect: the primary topic the sentiment is about (e.g., "shipping", "quality", "price", "customer_service", "packaging", "durability", "design", "ease_of_use", "battery", "fit", "taste", "value", "support", "warranty")
- confidence: a float between 0.0 and 1.0 indicating how confident you are in the classification
- reasoning: a brief one-sentence explanation justifying the sentiment and aspect choice

Sentiment guidelines:
- positive: the reviewer expresses satisfaction, praise, or happiness about the aspect
- negative: the reviewer expresses dissatisfaction, frustration, or disappointment about the aspect
- neutral: the reviewer states facts without clear emotional valence, or the sentiment is ambiguous
- mixed: the reviewer expresses BOTH positive and negative feelings in the same review (e.g., "great quality but terrible price")

Aspect guidelines:
- Choose the MOST SPECIFIC aspect the review primarily discusses
- If multiple aspects are mentioned, pick the one the reviewer emphasizes most
- Use lowercase snake_case for aspect names

Confidence guidelines:
- 0.9-1.0: sentiment and aspect are unambiguous and clearly stated
- 0.7-0.89: sentiment is fairly clear but requires minor inference
- 0.5-0.69: sentiment is somewhat ambiguous or the review is brief
- below 0.5: sentiment is very unclear or the review is contradictory

Output format:
Respond with ONLY a valid JSON object, no markdown fencing, no extra text.

Example output:
{{"sentiment": "positive", "aspect": "shipping", "confidence": 0.95, "reasoning": "Expresses strong satisfaction with delivery speed"}}

Review:
{input}"""

SYNTHETIC_TEMPLATES = [
    "The shipping was incredibly fast, arrived in just 2 days!",
    "Quality is terrible for the price, very disappointed.",
    "It's an okay product, nothing special but does what it says.",
    "Love the design but the battery life is atrocious, barely lasts 2 hours.",
    "Customer service was phenomenal — they replaced the item within 24 hours.",
    "The packaging was completely destroyed when it arrived, product was damaged.",
    "Best value for money I've found in this category, highly recommend.",
    "Returned it immediately. The size chart is completely wrong.",
    "Tastes exactly like the description, will definitely order again.",
    "The instructions are impossible to follow, took me 3 hours to assemble.",
    "Sturdy construction, this thing will last years. Very impressed.",
    "Waited 3 weeks for delivery and it still hasn't arrived.",
    "The material feels cheap but it actually works surprisingly well.",
    "Five stars! The noise cancellation on these headphones is incredible.",
    "Completely stopped working after just one week of normal use.",
    "It's fine. Does the job. Nothing to write home about.",
    "The warranty process was seamless, got a replacement in 5 days.",
    "Way too expensive for what you get. There are much better options.",
    "My kids absolutely love it, great gift idea for the holidays.",
    "The color in the photo looks nothing like what I received.",
    "Super easy to set up, was running in under 10 minutes.",
    "The flavor is bland and artificial, nothing like real fruit.",
    "Excellent build quality and the ergonomic design reduces wrist strain.",
    "Support team ghosted me after my initial complaint. Terrible experience.",
    "Perfect fit right out of the box, no adjustments needed.",
    "The app keeps crashing whenever I try to sync with the device.",
    "Good product overall but the price has gone up significantly since I last bought it.",
    "Lightweight and portable, perfect for traveling.",
    "The zipper broke on the second use. Cheap materials.",
    "Honestly exceeded my expectations, especially at this price point.",
    "Takes forever to charge and the battery indicator is inaccurate.",
    "The smell is overwhelming and doesn't fade even after washing.",
    "Fast delivery and everything was well-packaged. No complaints.",
    "Doesn't fit the description at all, feels like a knockoff.",
    "Great for beginners but advanced users will find it limiting.",
    "The customer support chat was helpful and resolved my issue in minutes.",
    "Screen arrived with a scratch right out of the box.",
    "Smooth texture and premium feel, worth every penny.",
    "The portion size is laughably small for the price they charge.",
    "Been using it daily for 6 months and it still works like new.",
]

CONFIG = UseCaseConfig(
    name="sentiment_aspect",
    display_name="Sentiment + Aspect Extraction",
    labels=["positive", "negative", "neutral", "mixed"],
    teacher_prompt=TEACHER_PROMPT,
    output_format="json",
    output_regex=None,
    teacher_input_tokens=450,
    student_input_tokens=60,
    teacher_output_tokens=40,
    student_output_tokens=30,
    synthetic_examples=1000,
    eval_samples=200,
    synthetic_input_templates=SYNTHETIC_TEMPLATES,
)
