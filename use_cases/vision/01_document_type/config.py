"""Document Type Classification — VLM single-label classification.

5 labels: invoice, receipt, letter, form, resume
Live demo use case: good public datasets, proven VLM classifier path.
"""

from shared.config import VisionUseCaseConfig

TEACHER_PROMPT = """\
You are an expert document classifier. Given a scanned document image, classify it into exactly one of these categories:

- invoice: bills, invoices, statements requesting payment, with line items, totals, and payment terms
- receipt: proof of purchase, transaction records, POS receipts, showing items bought and amount paid
- letter: formal or informal correspondence, memos, cover letters, with salutation and body text
- form: structured documents with fields to fill in, applications, questionnaires, tax forms, registration forms
- resume: curriculum vitae, job applications, professional summaries listing experience and education

Instructions:
1. Look at the overall layout and structure of the document.
2. Key visual cues:
   - Invoice: tables with amounts, "Invoice #", "Bill To", payment terms
   - Receipt: compact format, store name at top, item list with prices, total at bottom
   - Letter: letterhead, date, salutation ("Dear..."), body paragraphs, signature
   - Form: labeled fields, checkboxes, blank lines to fill in, section headers
   - Resume: name prominently displayed, sections like "Experience", "Education", "Skills"
3. If the document is ambiguous, classify by its PRIMARY purpose.
4. Ignore watermarks, stamps, or handwritten annotations — focus on the document type.

Output format:
Respond with EXACTLY one word: invoice, receipt, letter, form, or resume.

{input}"""

SYNTHETIC_TEMPLATES = [
    "A scanned invoice from Acme Corp with 5 line items totaling $3,420.00",
    "A grocery store receipt showing 12 items purchased at FreshMart",
    "A formal business letter from ABC Inc regarding a partnership proposal",
    "A job application form with fields for name, address, experience, and references",
    "A professional resume for a software engineer with 8 years of experience",
    "An invoice for consulting services rendered in December 2025",
    "A restaurant receipt from The Golden Spoon showing dinner for 4",
    "A cover letter addressed to the hiring manager at TechCorp",
    "A patient intake form from City Medical Center with health history fields",
    "A marketing coordinator resume highlighting campaign management skills",
    "A utility bill invoice from Pacific Gas & Electric for January service",
    "A hotel receipt showing 3-night stay with room service charges",
    "An internal memo regarding Q4 budget allocations",
    "A W-4 tax withholding form partially filled out",
    "A data scientist resume with publications and technical skills sections",
    "A freelance design invoice with hourly rates and project descriptions",
    "A pharmacy receipt showing prescription medications and copay amounts",
    "A recommendation letter from a university professor",
    "A vehicle registration renewal form from the DMV",
    "An executive resume for a VP of Engineering position",
    "A wholesale supplier invoice with quantity discounts applied",
    "A coffee shop receipt from a mobile payment terminal",
    "A formal complaint letter to a telecommunications company",
    "An employment application form for a retail position",
    "A graphic designer resume with portfolio links and certifications",
    "A monthly subscription invoice for cloud hosting services",
    "A dry cleaning receipt with itemized garment charges",
    "A thank-you letter following a business meeting",
    "An insurance claim form with accident details and witness information",
    "A nurse practitioner resume with clinical rotations listed",
    "An auto repair invoice with parts and labor breakdown",
    "An electronics store receipt showing a laptop purchase with warranty",
    "A legal demand letter regarding breach of contract",
    "A building permit application form for residential construction",
    "A project manager resume with PMP certification and agile experience",
    "A catering invoice for a corporate event with per-person pricing",
    "A gas station receipt showing fuel purchase and car wash",
    "A formal invitation letter to a charity gala event",
    "A scholarship application form requiring transcripts and essays",
    "A financial analyst resume with CFA designation and modeling skills",
]

CONFIG = VisionUseCaseConfig(
    name="document_type",
    display_name="Document Type Classification",
    labels=["invoice", "receipt", "letter", "form", "resume"],
    teacher_prompt=TEACHER_PROMPT,
    output_format="single_label",
    output_regex=r"\b(invoice|receipt|letter|form|resume)\b",
    teacher_input_tokens=800,
    student_input_tokens=200,
    teacher_output_tokens=2,
    student_output_tokens=1,
    synthetic_examples=500,
    eval_samples=100,
    synthetic_input_templates=SYNTHETIC_TEMPLATES,
    data_source="rvl_cdip",
)
