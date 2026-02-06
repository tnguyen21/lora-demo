"""PII Detection — span/entity extraction with JSON output.

6 labels: person, email, phone, ssn, address, credit_card
Extracts PII entities from business communications and returns structured JSON.
"""

from shared.config import UseCaseConfig

TEACHER_PROMPT = """\
You are an expert PII (Personally Identifiable Information) detection system.

Goal: Identify ALL PII entities in the given text and return them as a JSON array.

PII categories to detect:
- person: Full names, first+last name combinations. Include titles (Dr., Mr., Mrs.) as part of the name. Do NOT flag generic roles like "manager" or "team lead" — only actual named individuals.
- email: Email addresses in any standard format (user@domain.tld). Include the full address.
- phone: Phone numbers in any format — (555) 123-4567, 555-0123, +1-800-555-0199, etc. Include the full number as written.
- ssn: Social Security Numbers in any format — 123-45-6789, 123 45 6789, 123456789. Include the full number as written.
- address: Physical/mailing addresses. Include street address, city, state, zip when present. Capture the full address span as one entity.
- credit_card: Credit card numbers in any format — 4111-1111-1111-1111, 4111 1111 1111 1111, etc. Include the full number as written.

Instructions:
1. Scan the entire text carefully for ALL PII instances.
2. Extract the exact text span for each entity — do not paraphrase or normalize.
3. Assign the correct type from the 6 categories above.
4. If no PII is found, return an empty array: []
5. If the same entity appears multiple times, include it only once.
6. For ambiguous cases:
   - "John" alone is NOT a person (too common/ambiguous). "John Smith" IS a person.
   - Partial phone numbers (3-4 digits) are NOT phone numbers.
   - Company names are NOT person names.
   - Generic addresses like "our office" are NOT addresses. Only specific street/city addresses count.

Output format:
Respond with ONLY a valid JSON array of objects. Each object has two keys:
- "entity": the exact PII text span from the input
- "type": one of person, email, phone, ssn, address, credit_card

Example output:
[{{"entity": "Jane Doe", "type": "person"}}, {{"entity": "jane@example.com", "type": "email"}}]

Text to analyze:
{input}"""

SYNTHETIC_TEMPLATES = [
    "Hi team, please update the account for Michael Johnson at mjohnson@techcorp.com. His new phone is 555-0101.",
    "Send the NDA to Sarah Williams, sarah.w@lawfirm.io, at 742 Evergreen Terrace, Springfield, IL 62704.",
    "Customer complaint from David Lee (david.lee@shopmail.com): charged twice on card 4532-8810-1234-5678.",
    "Please process the refund for Emily Chen. Her SSN on file is 321-54-9876. Contact: echen@inbox.net.",
    "Meeting with Robert Garcia at our NYC office. Reach him at (212) 555-0198 or rgarcia@enterprise.org.",
    "Shipping label for Jessica Brown, 1600 Pennsylvania Ave NW, Washington, DC 20500. Phone: 202-555-0147.",
    "New hire paperwork: Thomas Anderson, SSN 456-78-9012, email t.anderson@matrix.dev, starting Monday.",
    "Invoice #4821 for Amanda Taylor. Billing address: 350 Fifth Avenue, New York, NY 10118. Card ending 4242.",
    "Urgent: contact Daniel Martinez at +1-415-555-0163 regarding the contract. His email is dmartinez@acme.co.",
    "Please verify the identity of Lisa Wang — SSN 789-01-2345, DOB on file. Reach her at lwang@corp.net.",
    "Forward the proposal to Christopher Davis, cdavis@bigco.com, and cc Mark Wilson at mwilson@bigco.com.",
    "Patient record update: Jennifer Lopez, 8800 Sunset Blvd, Los Angeles, CA 90069. Phone (310) 555-0142.",
    "Payroll adjustment for Kevin White. New direct deposit info. SSN: 234-56-7890. Email: kwhite@fin.org.",
    "Delivery confirmation to Laura Harris, 221B Baker Street, London. Contact: lharris@ukmail.co.uk, 555-0177.",
    "Please add James Miller to the vendor list. Reach him at jmiller@supplies.com or call 800-555-0126.",
    "Process credit card payment: 5425-2334-1098-7654, cardholder Patricia Robinson, billing zip 94105.",
    "IT ticket from Andrew Clark, aclark@devops.io: VPN access needed. His desk phone is ext 555-0134.",
    "Background check request for Maria Gonzalez, SSN 567-89-0123. Mail results to 456 Oak Lane, Austin, TX 78701.",
    "Schedule a call with Brian Thompson at brian.t@consult.com. Office: (312) 555-0189. Mobile: 555-0156.",
    "Tax form W-2 for Nicole Adams. SSN: 890-12-3456. Mailing: 789 Pine St, Denver, CO 80202.",
    "Hello, this is a reminder for Steven Wright to send his updated resume to hr@newjob.com by Friday.",
    "Customer support: Rebecca Moore reports unauthorized charges on card 4716-9012-3456-7890. Call her at 555-0193.",
    "Vendor onboarding: contact person is George Hall, george.hall@vendor.net, 1234 Commerce Blvd, Suite 100, Dallas, TX 75201.",
    "Please ship the replacement part to Karen Young at 567 Maple Dr, Portland, OR 97201. Her email is kyoung@home.com.",
    "Compliance filing for Edward King. SSN: 123-45-6789. Notify him at eking@legal.org or (617) 555-0112.",
    "Booking confirmation for Michelle Scott at mscott@travel.com. Loyalty card: 6011-0009-8765-4321. Phone: 555-0168.",
    "Expense report submitted by Ryan Turner, rturner@sales.io. Reimbursement to card ending 1234. Total: $847.",
    "Emergency contact update: primary is Stephanie Evans, phone (404) 555-0175, email sevans@family.net.",
    "New membership application from Timothy Baker, 321 Elm St, San Francisco, CA 94102. Call 415-555-0141.",
    "Referral bonus for Megan Phillips — refer questions to mphillips@partner.com or her cell 555-0187.",
    "Insurance claim #9921: claimant Dorothy Campbell, SSN 678-90-1234, address 900 Lake Shore Dr, Chicago, IL 60611.",
    "Contract renewal: point of contact is Jason Reed at jreed@logistics.com, direct line (469) 555-0129.",
    "Paycheck issue reported by Angela Cook. Her SSN is 345-67-8901, email acook@payroll.net. Please escalate.",
    "Invitation sent to Peter Morgan, pmorgan@events.com. Event at 200 Congress Ave, Austin, TX 78701. RSVP: 555-0195.",
    "Account verification: customer Linda Bell, card 3782-822463-10005, phone (702) 555-0113, lbell@casino.net.",
    "Updated contact for Samuel Howard: 555-0149, showard@engg.edu, office at 100 University Ave, Palo Alto, CA 94301.",
    "Please mail the certificate to Deborah Nelson at 450 Lexington Ave, New York, NY 10017. Email: dnelson@law.com.",
    "Fraud alert: transaction on card 4111-1111-1111-1111 by Rachel Green, rgreen@central.com. Call 555-0160.",
    "Benefits enrollment for Charles Murphy, SSN 901-23-4567. Confirm at cmurphy@hr.org or (503) 555-0138.",
    "Quarterly report distribution: send to Susan Cox at scox@board.com and copy Frank Ward, fward@board.com.",
]

CONFIG = UseCaseConfig(
    name="pii_detection",
    display_name="PII Detection",
    labels=["person", "email", "phone", "ssn", "address", "credit_card"],
    teacher_prompt=TEACHER_PROMPT,
    output_format="json",
    output_regex=None,
    teacher_input_tokens=500,
    student_input_tokens=60,
    teacher_output_tokens=50,
    student_output_tokens=30,
    synthetic_examples=1000,
    eval_samples=200,
    synthetic_input_templates=SYNTHETIC_TEMPLATES,
)
