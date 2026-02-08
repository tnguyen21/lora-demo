"""Async teacher prompting for data generation.

Text use cases: teacher labels via the Anthropic messages API (frontier model).
Vision use cases: teacher labels via Tinker (self-hosted VLM).
"""

import asyncio
import json
import os
import re

import anthropic
import tinker
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from shared.config import UseCaseConfig, VisionUseCaseConfig
from shared.output_parsers import parse_json_output, parse_single_label


def setup_vision_clients(config: VisionUseCaseConfig):
    """Create service client, sampling client, tokenizer, renderer, and image_processor for vision tasks."""
    from tinker_cookbook.image_processing_utils import get_image_processor

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=config.teacher_model)
    tokenizer = get_tokenizer(config.teacher_model)
    image_processor = get_image_processor(config.teacher_model)
    renderer = renderers.get_renderer(config.renderer_name, tokenizer, image_processor)

    return sampling_client, tokenizer, renderer, image_processor


def _parse_teacher_response(response: str, config: UseCaseConfig) -> str | None:
    """Parse teacher response based on output format."""
    if config.output_format == "single_label":
        return parse_single_label(response, config.labels, config.output_regex)
    elif config.output_format == "json":
        parsed = parse_json_output(response)
        return json.dumps(parsed) if parsed is not None else None
    elif config.output_format == "free_text":
        if config.output_regex:
            match = re.search(config.output_regex, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        return response.strip() if response.strip() else None
    return None


def _setup_anthropic_client() -> anthropic.AsyncAnthropic:
    """Create an async Anthropic client (reads ANTHROPIC_API_KEY from env)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    return anthropic.AsyncAnthropic()


async def generate_teacher_labels(
    config: UseCaseConfig,
    inputs: list[str],
    output_path: str,
) -> list[dict]:
    """Generate teacher labels for text inputs via the Anthropic API.

    Args:
        config: Use case configuration with teacher prompt and model settings.
        inputs: List of raw text inputs to label.
        output_path: Path to write the JSONL training data.

    Returns:
        List of {input, output} dicts for all successfully labeled examples.
    """
    client = _setup_anthropic_client()
    semaphore = asyncio.Semaphore(10)

    async def sample_one(text: str) -> tuple[str, str | None]:
        prompt_text = config.teacher_prompt.format(input=text)
        async with semaphore:
            message = await client.messages.create(
                model=config.teacher_model,
                max_tokens=config.teacher_max_tokens,
                temperature=config.teacher_temperature,
                messages=[{"role": "user", "content": prompt_text}],
            )
        response = message.content[0].text
        parsed = _parse_teacher_response(response, config)
        return (text, parsed)

    results = []
    for coro in tqdm_asyncio.as_completed([sample_one(inp) for inp in inputs], total=len(inputs)):
        text, label = await coro
        if label is not None:
            results.append({"input": text, "output": label})

    # Save as JSONL in short message format (no teacher prompt)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            record = {
                "messages": [
                    {"role": "user", "content": r["input"]},
                    {"role": "assistant", "content": r["output"]},
                ]
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)}/{len(inputs)} labeled examples to {output_path}")
    return results


async def generate_teacher_labels_vlm(
    config: VisionUseCaseConfig,
    image_paths: list[str],
    output_path: str,
) -> list[dict]:
    """Generate teacher labels for image inputs using async VLM sampling.

    Args:
        config: Vision use case configuration.
        image_paths: List of paths to image files.
        output_path: Path to write the JSONL training data.

    Returns:
        List of {input, output} dicts for all successfully labeled examples.
    """
    from PIL import Image
    from tinker_cookbook.image_processing_utils import resize_image
    from tinker_cookbook.renderers import ImagePart, Message, TextPart

    sampling_client, tokenizer, renderer, _ = setup_vision_clients(config)

    params = tinker.SamplingParams(
        max_tokens=config.teacher_max_tokens,
        temperature=config.teacher_temperature,
        stop=renderer.get_stop_sequences(),
    )

    async def sample_one(image_path: str) -> tuple[str, str | None]:
        pil_image = Image.open(image_path)
        pil_image = resize_image(image=pil_image, max_size=config.max_image_size)

        prompt_text = config.teacher_prompt.format(input="")
        messages = [
            Message(
                role="user",
                content=[
                    ImagePart(type="image", image=pil_image),
                    TextPart(type="text", text=prompt_text),
                ],
            ),
        ]
        model_input = renderer.build_generation_prompt(messages=messages, role="assistant")
        result = await sampling_client.sample_async(prompt=model_input, sampling_params=params, num_samples=1)
        response = tokenizer.decode(result.sequences[0].tokens)
        parsed = _parse_teacher_response(response, config)
        return (image_path, parsed)

    results = []
    for coro in tqdm_asyncio.as_completed([sample_one(p) for p in image_paths], total=len(image_paths)):
        path, label = await coro
        if label is not None:
            results.append({"input": path, "output": label})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            record = {
                "messages": [
                    {"role": "user", "content": r["input"]},
                    {"role": "assistant", "content": r["output"]},
                ]
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)}/{len(image_paths)} labeled examples to {output_path}")
    return results
