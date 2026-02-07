"""Self-host the fine-tuned ticket routing model (merge, serve, test)."""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.import_utils import load_config

CONFIG = load_config(__file__)

DEFAULT_ADAPTER_DIR = "/tmp/tinker-weights/ticket_routing"
DEFAULT_MERGED_DIR = "/tmp/tinker-merged/ticket_routing"
DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = "ticket-routing"
TEST_CASES = [
    ("I was double-charged on my last invoice.", "billing"),
    ("API calls returning 500 errors since the update.", "technical"),
    ("Can't log into my account after password reset.", "account"),
    ("Do you plan to add dark mode?", "general"),
]


def load_merge_script():
    import importlib.util

    def load_from_path(script_path: Path):
        spec = importlib.util.spec_from_file_location("tinker_merge_adapter_script", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        import tinker_cookbook

        package_root = Path(tinker_cookbook.__file__).resolve().parent
        script_path = package_root / "scripts" / "merge_tinker_adapter_to_hf_model.py"
        if script_path.exists():
            return load_from_path(script_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tinker-cookbook is not installed. Install it with `pip install tinker-cookbook` or `uv add tinker-cookbook`."
        ) from exc

    raise FileNotFoundError("merge_tinker_adapter_to_hf_model.py not found in the installed tinker-cookbook package. Try upgrading it.")


def validate_adapter_dir(adapter_dir: Path) -> None:
    required = ["adapter_model.safetensors", "adapter_config.json"]
    missing = [name for name in required if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing adapter files in {adapter_dir}: {', '.join(missing)}")


def merge_adapter(adapter_dir: Path, output_dir: Path) -> None:
    validate_adapter_dir(adapter_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output dir {output_dir} already exists and is not empty.")
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_script = load_merge_script()

    merge_script.log("Loading HF Model")
    hf_model = merge_script.load_model(CONFIG.student_model)

    merge_script.log("Loading Adapter Weights")
    adapter_weights, adapter_config = merge_script.load_adapter_weights(str(adapter_dir))

    merge_script.log("Merging Adapter Weights")
    merge_script.merge_adapter_weights(hf_model, adapter_weights, adapter_config)

    merge_script.log("Saving Merged Model")
    hf_model.save_pretrained(str(output_dir))
    tokenizer = merge_script.AutoTokenizer.from_pretrained(CONFIG.student_model)
    tokenizer.save_pretrained(str(output_dir))
    merge_script.log(f"Merged model saved to {output_dir}")


def print_serve_commands(model_dir: Path, adapter_dir: Path | None) -> None:
    merged_cmd = f"vllm serve {model_dir} --served-model-name {DEFAULT_MODEL} --max-model-len 2048 --dtype bfloat16"

    print("Merged model serving command (vLLM):")
    print(merged_cmd)
    print()

    adapter_dir_display = adapter_dir if adapter_dir else "<adapter-dir>"
    lora_cmd = (
        f"vllm serve {CONFIG.student_model} "
        "--enable-lora "
        f"--lora-modules ticket_routing={adapter_dir_display} "
        f"--served-model-name {DEFAULT_MODEL} "
        "--max-model-len 2048 "
        "--dtype bfloat16"
    )

    print("Serve LoRA directly (no merge required):")
    print(lora_cmd)


def extract_label(text: str) -> str:
    cleaned = re.sub(r"<\|.*?\|>", "", text).strip()
    match = re.search(r"Final Answer:\s*(\w+)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    if not cleaned:
        return "???"
    return cleaned.split()[0].strip().lower()


def send_chat_completion(base_url: str, model: str, message: str) -> str:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.0,
        "max_tokens": 12,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"]


def test_server(base_url: str, model: str) -> None:
    print(f"Testing {base_url} (model={model})")
    for text, expected in TEST_CASES:
        try:
            response = send_chat_completion(base_url, model, text)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Request failed: {exc}") from exc

        predicted = extract_label(response)
        status = "OK" if predicted == expected else "MISS"
        print(f"[{status}] expected={expected} predicted={predicted}")
        print(f"  input: {text}")
        print(f"  response: {response.strip()}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge LoRA into base model")
    merge_parser.add_argument(
        "--adapter-dir",
        default=DEFAULT_ADAPTER_DIR,
        help=f"Adapter path (default: {DEFAULT_ADAPTER_DIR})",
    )
    merge_parser.add_argument(
        "--output-dir",
        default=DEFAULT_MERGED_DIR,
        help=f"Merged model output dir (default: {DEFAULT_MERGED_DIR})",
    )

    serve_parser = subparsers.add_parser("serve", help="Print vLLM commands")
    serve_parser.add_argument(
        "--model-dir",
        default=DEFAULT_MERGED_DIR,
        help=f"Merged model path (default: {DEFAULT_MERGED_DIR})",
    )
    serve_parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional adapter path for the LoRA serve command",
    )

    test_parser = subparsers.add_parser("test", help="Send test requests")
    test_parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL for the OpenAI-compatible server (default: {DEFAULT_URL})",
    )
    test_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to send in requests (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()

    if args.command == "merge":
        merge_adapter(Path(os.path.expanduser(args.adapter_dir)), Path(os.path.expanduser(args.output_dir)))
    elif args.command == "serve":
        adapter_dir = Path(os.path.expanduser(args.adapter_dir)) if args.adapter_dir else None
        print_serve_commands(Path(os.path.expanduser(args.model_dir)), adapter_dir)
    elif args.command == "test":
        test_server(args.url, args.model)


if __name__ == "__main__":
    main()
