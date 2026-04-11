import torch
from datetime import datetime
from pathlib import Path
from src.model import load_model, get_model_name_slug
from src.checkpoint import load_checkpoint
from src.config import CHECKPOINT_PATH, EVAL_RESULTS_DIR
from src.tokenizer import load_tokenizer
from src.baseline import evaluate_baseline, save_results


def run_evaluate(device: torch.device):
    checkpoint = load_checkpoint(path=CHECKPOINT_PATH, device=device)
    model_state_dict = checkpoint["model_state_dict"]
    model_name = checkpoint["model_name"]
    dataset_name = checkpoint["dataset_name"]
    learning_rate = checkpoint["learning_rate"]
    batch_size = checkpoint["batch_size"]
    print(f"##### Evaluating model from checkpoint {CHECKPOINT_PATH} #####")
    model = load_model(model_name=model_name)
    model.load_state_dict(model_state_dict)
    model.to(device)
    tokenizer = load_tokenizer(model_name=model_name)
    results = evaluate_baseline(model, tokenizer)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    model_name_slug = get_model_name_slug(model_name)
    save_results(
        results=results,
        pad_token_id=tokenizer.eos_token_id,
        path=(
            Path(EVAL_RESULTS_DIR)
            / "post_sft"
            / f"{model_name_slug}-{dataset_name}-{batch_size}-{learning_rate}-{timestamp}.json"
        ),
        model_name=model_name,
        timestamp=timestamp,
    )
    print("Evaluation completed and saved.")
