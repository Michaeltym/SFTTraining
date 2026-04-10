import json
import torch
from pathlib import Path
from datetime import datetime

from src.model import load_model
from src.tokenizer import load_tokenizer
from src.eval_prompts import EVAL_PROMPTS
from src.config import (
    EVAL_RESULTS_DIR,
    PRETRAINED_MODEL,
    MAX_NEW_TOKEN,
    USE_CHAT_TEMPLATE,
)


def run_baseline(device: torch.device):
    model = load_model()
    model.to(device)
    tokenizer = load_tokenizer()
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
    save_baseline(results, pad_token_id)


def save_baseline(results: dict[str, list[dict[str, str]]], pad_token_id: str):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = (
        Path(EVAL_RESULTS_DIR)
        / f"baseline-{timestamp}{'-with-chat-template' if USE_CHAT_TEMPLATE else ''}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": PRETRAINED_MODEL,
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
        print("Baseline evaluation completed and saved.")
