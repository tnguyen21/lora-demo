"""Support Ticket Routing — single-label classification.

4 labels: billing, technical, account, general
Live demo use case: universally relatable, simple classification.
"""

from shared.config import UseCaseConfig

TEACHER_PROMPT = """\
You are an expert support ticket router for a SaaS company.

Goal: Classify the customer support ticket into exactly one of these categories:
- billing: payment issues, invoices, charges, refunds, subscription changes, pricing questions
- technical: bugs, errors, performance issues, API problems, integration help, feature not working
- account: login issues, password resets, account settings, permissions, profile updates, account deletion
- general: feature requests, feedback, general questions, onboarding help, documentation questions

Instructions:
1. Read the ticket carefully. Focus on the primary intent, not secondary mentions.
2. If the ticket mentions multiple issues, classify by the MAIN ask (what they want resolved).
3. Billing vs Account: if it's about money/charges, it's billing. If it's about access/settings, it's account.
4. Technical vs General: if something is broken/not working, it's technical. If they're asking how to do something, it's general.
5. When in doubt between categories, prefer the more specific one.

Output format:
Respond with EXACTLY one word: billing, technical, account, or general.

Ticket:
{input}"""

SYNTHETIC_TEMPLATES = [
    "I was double-charged on my last invoice for the Pro plan. Can you refund the extra charge?",
    "My API calls are returning 500 errors since the last update. Here are the logs.",
    "I can't log into my account. It says my password is incorrect but I just changed it yesterday.",
    "Do you have any plans to add a dark mode to the dashboard?",
    "I need to upgrade from the Starter plan to Enterprise. What's the pricing?",
    "The export feature is broken — it generates empty CSV files every time I try.",
    "I want to add a new team member to our organization but can't find the invite button.",
    "How do I set up the webhook integration with Slack?",
    "Please cancel my subscription effective immediately and process a prorated refund.",
    "The search function takes 30+ seconds to return results. It used to be instant.",
    "I need to update the billing email address on our account.",
    "We're seeing intermittent timeout errors on the /v2/users endpoint.",
    "Can you help me understand the difference between the Team and Business plans?",
    "I accidentally deleted my project. Is there a way to recover it?",
    "Our SSO integration with Okta stopped working after your maintenance window.",
    "I'd like to request a copy of all invoices from the past 12 months.",
    "The mobile app crashes every time I try to open the settings page.",
    "How do I transfer ownership of my organization to another admin?",
    "Is there an API rate limit? I'm getting 429 errors during peak hours.",
    "I'm a new customer and would love a walkthrough of the main features.",
    "You charged my credit card twice this month. Transaction IDs: TXN-4821 and TXN-4822.",
    "The dashboard shows stale data — metrics haven't updated in 3 hours.",
    "I need to change my account email from john@old.com to john@new.com.",
    "What's the recommended way to handle pagination in your REST API?",
    "Can you apply the nonprofit discount to our billing account?",
    "File uploads fail silently when the file is over 10MB. No error message shown.",
    "I want to enable two-factor authentication but the option is grayed out.",
    "We love the product! Any chance you'll add Jira integration soon?",
    "My trial expired but I wasn't able to enter payment info before it ended.",
    "The real-time notifications feature sends duplicate alerts for every event.",
    "How do I revoke API keys for a former employee?",
    "Can someone walk me through the onboarding checklist? I'm stuck on step 3.",
    "I'm being charged for 50 seats but we only have 30 active users.",
    "The PDF report generator produces garbled text for non-English characters.",
    "My account was locked after too many failed login attempts. How do I unlock it?",
    "What's the SLA for the Enterprise plan? I need to present this to my CTO.",
    "I need a refund for the annual payment — we're switching providers.",
    "Webhooks are being delivered with a 15-minute delay. This used to be near-instant.",
    "Can I merge two separate accounts into one organization?",
    "Your documentation on the batch API is outdated — the endpoints have changed.",
]

CONFIG = UseCaseConfig(
    name="ticket_routing",
    display_name="Support Ticket Routing",
    labels=["billing", "technical", "account", "general"],
    teacher_prompt=TEACHER_PROMPT,
    output_format="single_label",
    output_regex=r"\b(billing|technical|account|general)\b",
    teacher_input_tokens=350,
    student_input_tokens=40,
    teacher_output_tokens=2,
    student_output_tokens=1,
    synthetic_examples=1000,
    eval_samples=200,
    synthetic_input_templates=SYNTHETIC_TEMPLATES,
)
