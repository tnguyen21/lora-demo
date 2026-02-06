"""
Sample from the prompt-distillation fine-tuned model to test language classification.

The model was trained via prompt distillation: a teacher (Qwen3-30B-A3B) classified
languages using a detailed prompt, and the student was trained on just
(sentence -> language_code) pairs. So we test with raw sentences and expect
two-letter language codes back.

Usage:
    python scripts/sample_test_model.py
"""

import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
MODEL_PATH = "tinker://71778cb9-927b-5790-b212-d24b04926472:train:0/sampler_weights/final"

# Test sentences in various languages with expected labels
TEST_SENTENCES = [
    ("And he said, Mama, I'm home.", "en"),
    ("Et il a dit, maman, je suis à la maison.", "fr"),
    ("Y él dijo: Mamá, estoy en casa.", "es"),
    ("und er hat gesagt, Mama ich bin daheim.", "de"),
    ("他说，妈妈，我回来了。", "zh"),
    ("И он сказал: Мама, я дома.", "ru"),
    ("और उसने कहा, माँ, मैं घर आया हूं।", "hi"),
    ("Ve Anne, evdeyim dedi.", "tr"),
    ("Và anh ấy nói, Mẹ, con đã về nhà.", "vi"),
    ("وقال، ماما، لقد عدت للمنزل.", "ar"),
]


def main():
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {MODEL_PATH}")
    print()

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=MODEL_PATH)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer("qwen3", tokenizer)

    params = tinker.SamplingParams(
        max_tokens=20,
        temperature=0.15,
        stop=renderer.get_stop_sequences(),
    )

    correct = 0
    total = len(TEST_SENTENCES)

    for sentence, expected in TEST_SENTENCES:
        # Match the training data format: user message is the raw sentence,
        # model should respond with just the language code.
        messages = [{"role": "user", "content": sentence}]
        model_input = renderer.build_generation_prompt(messages)

        result = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=params,
        ).result()

        response = tokenizer.decode(result.sequences[0].tokens).strip()
        # Strip special tokens that may appear in decoded output
        response = re.sub(r"<\|.*?\|>", "", response).strip()

        # The model should output just the language code, but try to extract it
        # if it outputs something longer
        predicted = response.split()[0].strip().lower() if response else "???"
        # Also try regex for "Final Answer: xx" format just in case
        match = re.search(r"Final Answer:\s*(\w+)", response)
        if match:
            predicted = match.group(1)

        is_correct = predicted == expected
        correct += is_correct
        status = "OK" if is_correct else "MISS"

        print(f"[{status}] expected={expected} predicted={predicted}")
        print(f"  input: {sentence[:60]}")
        print(f"  response: {response[:120]}")
        print()

    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.0f}%)")


if __name__ == "__main__":
    main()
