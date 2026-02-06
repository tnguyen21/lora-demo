"""Template: Vision Classification — Fill in the TODOs to create your own VLM use case.

Copy this directory and customize for your task.
"""

from shared.config import VisionUseCaseConfig

# TODO: Define your teacher prompt for the VLM.
# Include visual cues and distinguishing features for each class.
# Use {input} as the placeholder.
TEACHER_PROMPT = """\
You are an expert image classifier.

Goal: Classify the provided image into exactly one of these categories:
# TODO: List your categories with visual descriptions

Instructions:
# TODO: Add task-specific instructions about visual features to look for

Output format:
Respond with EXACTLY one word from the label set.

{input}"""

# TODO: Add synthetic input templates (descriptions of what the images contain)
SYNTHETIC_TEMPLATES = [
    # TODO: "Description of an image that should be classified as label_a",
    # TODO: "Description of an image that should be classified as label_b",
]

# TODO: Customize this config for your use case
CONFIG = VisionUseCaseConfig(
    name="my_vlm_classifier",  # TODO: Change to your task name
    display_name="My Vision Classifier",  # TODO: Change to your display name
    labels=["label_a", "label_b", "label_c"],  # TODO: Your labels
    teacher_prompt=TEACHER_PROMPT,
    output_format="single_label",
    output_regex=r"\b(label_a|label_b|label_c)\b",  # TODO: Update regex
    # TODO: Estimate token counts for your task (vision tasks use more input tokens)
    teacher_input_tokens=800,
    student_input_tokens=200,
    teacher_output_tokens=2,
    student_output_tokens=1,
    synthetic_examples=500,
    eval_samples=100,
    synthetic_input_templates=SYNTHETIC_TEMPLATES,
    data_source=None,  # TODO: Set to HuggingFace dataset name if applicable
)
