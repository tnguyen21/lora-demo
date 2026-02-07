"""Self-host the fine-tuned ticket routing model (merge, serve, test)."""

import argparse
import json
import os
import re
import sys
import subprocess
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


def validate_adapter_dir(adapter_dir: Path) -> None:
    required = ["adapter_model.safetensors", "adapter_config.json"]
    missing = [name for name in required if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing adapter files in {adapter_dir}: {', '.join(missing)}")


def merge_adapter(adapter_dir: Path, output_dir: Path) -> None:
    try:
        import json
        from datetime import datetime

        import torch
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing merge dependencies. Install with `pip install torch safetensors transformers`.") from exc

    def log(message: str) -> None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def load_model(model_path: str):
        return AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16)

    def load_adapter_weights(adapter_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = load_file(os.path.expanduser(adapter_path + "/adapter_model.safetensors"), device=device)
        with open(os.path.expanduser(adapter_path + "/adapter_config.json"), "r") as f:
            config = json.load(f)
        return weights, config

    def apply_merged_weight(target: torch.Tensor, merged_lora: torch.Tensor) -> None:
        assert target.shape == merged_lora.shape, (target.shape, merged_lora.shape)
        new_data = target.float() + merged_lora.float().to(target.device)
        target.copy_(new_data.to(target.dtype))

    def merge_adapter_weights(base_model, adapter_weights: dict[str, torch.Tensor], config: dict) -> None:
        scaling = config["lora_alpha"] / config["r"]
        adapter_weight_names = [n.replace(".lora_A", "") for n in adapter_weights if ".lora_A" in n]

        model_state_dict = base_model.state_dict()
        is_gpt_oss = "GptOss" in str(type(base_model))

        for name in adapter_weight_names:
            target_key = name.replace("base_model.model.", "").replace("model.unembed_tokens", "lm_head")
            lora_a = adapter_weights[name.replace(".weight", ".lora_A.weight")].float()
            lora_b = adapter_weights[name.replace(".weight", ".lora_B.weight")].float() * scaling
            if ".experts" not in name:
                if is_gpt_oss:
                    target_key = target_key.replace(".attn", ".self_attn")
                assert target_key in model_state_dict, (name, target_key)
                merged_lora = torch.nn.functional.linear(lora_a.T, lora_b).T
                assert merged_lora.shape == model_state_dict[target_key].shape, (
                    name,
                    merged_lora.shape,
                    model_state_dict[target_key].shape,
                )
                apply_merged_weight(model_state_dict[target_key], merged_lora)
            else:
                assert len(lora_a.shape) == 3 and len(lora_b.shape) == 3, (lora_a.shape, lora_b.shape)
                if lora_a.shape[0] == 1:
                    assert lora_b.shape[0] > 1
                    lora_a = lora_a.expand(lora_b.shape[0], -1, -1)
                elif lora_b.shape[0] == 1:
                    assert lora_a.shape[0] > 1
                    lora_b = lora_b.expand(lora_a.shape[0], -1, -1)
                merged_lora = torch.bmm(lora_a.transpose(-1, -2), lora_b.transpose(-1, -2))

                target_key = target_key.replace(".w1.weight", ".gate_proj.weight")
                target_key = target_key.replace(".w3.weight", ".up_proj.weight")
                target_key = target_key.replace(".w2.weight", ".down_proj.weight")

                if not is_gpt_oss:
                    merged_lora = merged_lora.transpose(-1, -2)
                    for exp_idx in range(merged_lora.shape[0]):
                        target_key_exp = target_key.replace(".experts", f".experts.{exp_idx}")
                        assert target_key_exp in model_state_dict, (name, target_key_exp)
                        assert merged_lora[exp_idx].shape == model_state_dict[target_key_exp].shape, (
                            target_key_exp,
                            merged_lora[exp_idx].shape,
                            model_state_dict[target_key_exp].shape,
                        )
                        apply_merged_weight(model_state_dict[target_key_exp], merged_lora[exp_idx])
                else:
                    if target_key.endswith((".gate_proj.weight", ".up_proj.weight")):
                        target_key = target_key.replace(".gate_proj.weight", ".gate_up_proj").replace(".up_proj.weight", ".gate_up_proj")
                        idx = 0 if target_key.endswith(".gate_up_proj") else 1
                        target = model_state_dict[target_key][:, :, idx::2]
                    else:
                        target_key = target_key.replace(".down_proj.weight", ".down_proj")
                        target = model_state_dict[target_key]
                    apply_merged_weight(target, merged_lora)

    validate_adapter_dir(adapter_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output dir {output_dir} already exists and is not empty.")
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading HF Model")
    hf_model = load_model(CONFIG.student_model)

    log("Loading Adapter Weights")
    adapter_weights, adapter_config = load_adapter_weights(str(adapter_dir))

    log("Merging Adapter Weights")
    merge_adapter_weights(hf_model, adapter_weights, adapter_config)

    log("Saving Merged Model")
    hf_model.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.student_model)
    tokenizer.save_pretrained(str(output_dir))
    log(f"Merged model saved to {output_dir}")


def serve_model(model_dir: Path, adapter_dir: Path | None, mode: str, print_only: bool) -> None:
    merged_cmd = [
        "vllm",
        "serve",
        str(model_dir),
        "--served-model-name",
        DEFAULT_MODEL,
        "--max-model-len",
        "2048",
        "--dtype",
        "bfloat16",
    ]

    adapter_dir_display = adapter_dir if adapter_dir else "<adapter-dir>"
    lora_cmd = [
        "vllm",
        "serve",
        CONFIG.student_model,
        "--enable-lora",
        "--lora-modules",
        f"ticket_routing={adapter_dir_display}",
        "--served-model-name",
        DEFAULT_MODEL,
        "--max-model-len",
        "2048",
        "--dtype",
        "bfloat16",
    ]

    if mode == "merged":
        cmd = merged_cmd
    elif mode == "lora":
        if not adapter_dir:
            raise ValueError("--adapter-dir is required when --mode lora is selected.")
        cmd = lora_cmd
    else:
        raise ValueError(f"Unknown serve mode: {mode}")

    print("Merged model serving command (vLLM):")
    print(" ".join(merged_cmd))
    print()
    print("Serve LoRA directly (no merge required):")
    print(" ".join(lora_cmd))
    print()
    print("Launching vLLM server:")
    print(" ".join(cmd))

    if print_only:
        return

    subprocess.run(cmd, check=True)


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
    serve_parser.add_argument(
        "--mode",
        choices=["merged", "lora"],
        default="merged",
        help="Which serving mode to launch (default: merged)",
    )
    serve_parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the command; do not start vLLM",
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
        serve_model(
            Path(os.path.expanduser(args.model_dir)),
            adapter_dir,
            mode=args.mode,
            print_only=args.print_only,
        )
    elif args.command == "test":
        test_server(args.url, args.model)


if __name__ == "__main__":
    main()
