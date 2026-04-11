import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Any

from src.model import get_model_name_slug, load_model
from src.tokenizer import load_tokenizer
from src.eval_prompts import EVAL_PROMPTS
from src.config import (
    EVAL_RESULTS_DIR,
    MODEL_NAME,
    MAX_NEW_TOKEN,
    USE_CHAT_TEMPLATE,
)


def evaluate_baseline(model: Any, tokenizer: Any):
    results: dict[str, list[dict[str, str]]] = {}
    pad_token_id = tokenizer.eos_token_id
    for category, prompts in EVAL_PROMPTS.items():
        results[category] = []
        for prompt in prompts:
            if USE_CHAT_TEMPLATE:
                messages = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKEN,
                    pad_token_id=pad_token_id,
                )
                output_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )

            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKEN,
                    pad_token_id=pad_token_id,
                )
                output_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
            results[category].append({"prompt": prompt, "output": output_text})
    return results


def run_baseline(device: torch.device):
    model = load_model(model_name=MODEL_NAME)
    model.to(device)
    tokenizer = load_tokenizer(model_name=MODEL_NAME)
    results = evaluate_baseline(model, tokenizer)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    model_name_slug = get_model_name_slug(MODEL_NAME)
    save_results(
        results=results,
        pad_token_id=tokenizer.eos_token_id,
        path=(
            Path(EVAL_RESULTS_DIR) / "baseline" / f"{model_name_slug}-{timestamp}.json"
        ),
        model_name=MODEL_NAME,
        timestamp=timestamp,
    )
    print("Baseline evaluation completed and saved.")


def save_results(
    results: dict[str, list[dict[str, str]]],
    pad_token_id: int,
    path: Path,
    model_name: str,
    timestamp: str,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "evaluated_at": timestamp,
                "generation_config": {
                    "max_new_token": MAX_NEW_TOKEN,
                    "pad_token_id": pad_token_id,
                    "use_chat_template": USE_CHAT_TEMPLATE,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
