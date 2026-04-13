import torch
from datetime import datetime
from pathlib import Path
from src.model import get_model_name_slug
from src.checkpoint import load_checkpoint
from src.config import CHECKPOINT_PATH, EVAL_RESULTS_DIR
from src.baseline import evaluate_baseline, save_results
from src.runtime import build_checkpoint_runtime


def run_evaluate(device: torch.device):
    checkpoint = load_checkpoint(path=CHECKPOINT_PATH, device=device)
    model_name = checkpoint["model_name"]
    dataset_name = checkpoint["dataset_name"]
    learning_rate = checkpoint["learning_rate"]
    batch_size = checkpoint["batch_size"]
    print(
        f"##### Evaluating model from checkpoint {CHECKPOINT_PATH} for model {model_name} on dataset {dataset_name} #####"
    )
    model, tokenizer = build_checkpoint_runtime(checkpoint=checkpoint, device=device)
    results = evaluate_baseline(model, tokenizer)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    model_name_slug = get_model_name_slug(model_name)
    save_results(
        results=results,
        pad_token_id=tokenizer.eos_token_id,
        path=(
            Path(EVAL_RESULTS_DIR)
            / "post_sft"
            / f"{dataset_name}-{model_name_slug}-{batch_size}-{learning_rate}-{timestamp}.json"
        ),
        model_name=model_name,
        timestamp=timestamp,
    )
    print("Evaluation completed and saved.")
