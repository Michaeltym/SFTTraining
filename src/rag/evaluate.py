import json
import torch
from datetime import datetime
from pathlib import Path

from src.checkpoint import load_checkpoint
from src.config import CHECKPOINT_PATH, EVAL_RESULTS_DIR
from src.eval_prompts import RAG_REGRESSION_ITEMS
from src.model import get_model_name_slug
from src.rag.inference import InferenceResult, run_inference
from src.runtime import build_checkpoint_runtime


def run_rag_evaluate(device: torch.device):
    questions = [item["prompt"] for item in RAG_REGRESSION_ITEMS]

    checkpoint = load_checkpoint(path=CHECKPOINT_PATH, device=device)
    model_name = checkpoint["model_name"]
    dataset_name = checkpoint["dataset_name"]
    learning_rate = checkpoint["learning_rate"]
    batch_size = checkpoint["batch_size"]

    model, tokenizer = build_checkpoint_runtime(checkpoint=checkpoint, device=device)
    results: list[InferenceResult] = []

    for query in questions:
        results.append(run_inference(model=model, tokenizer=tokenizer, query=query))

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    model_name_slug = get_model_name_slug(model_name)
    save_rag_results(
        results=results,
        model_name=model_name,
        dataset_name=dataset_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        timestamp=timestamp,
        path=(
            Path(EVAL_RESULTS_DIR)
            / "rag"
            / f"{dataset_name}-{model_name_slug}-{batch_size}-{learning_rate}-{timestamp}.json"
        ),
    )
    print("RAG evaluation completed and saved.")


def save_rag_results(
    results: list[InferenceResult],
    model_name: str,
    dataset_name: str,
    batch_size: int,
    learning_rate: float,
    timestamp: str,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "rag_evaluate",
                "model": model_name,
                "dataset": dataset_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "evaluated_at": timestamp,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
