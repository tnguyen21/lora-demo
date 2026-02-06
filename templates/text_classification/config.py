"""Template: Text Classification — Fill in the TODOs to create your own use case.

Copy this directory and customize for your task.
"""

from shared.config import UseCaseConfig

# TODO: Define your teacher prompt. This is the detailed rubric that makes a
# large model reliable at your task. Include:
# - Clear label definitions
# - Edge case handling
# - Output format instructions
# Use {input} as the placeholder for the text to classify.
TEACHER_PROMPT = """\
You are an expert classifier.

Goal: Classify the provided text into exactly one of these categories:
# TODO: List your categories with descriptions

Instructions:
# TODO: Add task-specific instructions, edge cases, disambiguation rules

Output format:
Respond with EXACTLY one word from the label set.

Text to classify:
{input}"""

# TODO: Add 30-40 synthetic input templates for data generation.
# These should cover the full range of inputs your model will see.
# Use {idx} as a placeholder for a unique index if needed.
SYNTHETIC_TEMPLATES = [
    # TODO: "Example input text that should be classified as label_a",
    # TODO: "Example input text that should be classified as label_b",
]

# TODO: Customize this config for your use case
CONFIG = UseCaseConfig(
    name="my_classifier",  # TODO: Change to your task name
    display_name="My Text Classifier",  # TODO: Change to your display name
    labels=["label_a", "label_b", "label_c"],  # TODO: Your labels
    teacher_prompt=TEACHER_PROMPT,
    output_format="single_label",
    output_regex=r"\b(label_a|label_b|label_c)\b",  # TODO: Update regex to match your labels
    # TODO: Estimate token counts for your task
    teacher_input_tokens=500,
    student_input_tokens=40,
    teacher_output_tokens=2,
    student_output_tokens=1,
    synthetic_examples=1000,
    eval_samples=200,
    synthetic_input_templates=SYNTHETIC_TEMPLATES,
)
