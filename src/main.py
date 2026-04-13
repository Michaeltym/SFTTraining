import torch
from src.baseline import run_baseline
from src.training import run_training_loop
from src.resume import run_resume
from src.evaluate import run_evaluate
from src.config import (
    MODE,
    MODE_BASELINE,
    MODE_RESUME,
    MODE_TRAIN,
    MODE_EVALUATE,
    MODE_INFERENCE,
)
from src.rag.inference import run_inference, print_inference_result
from src.runtime import load_checkpoint_runtime


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    device = torch.device(device)

    if MODE == MODE_BASELINE:
        run_baseline(device=device)
    elif MODE == MODE_TRAIN:
        run_training_loop(device=device)
    elif MODE == MODE_RESUME:
        run_resume(device=device)
    elif MODE == MODE_EVALUATE:
        run_evaluate(device=device)
    elif MODE == MODE_INFERENCE:
        questions = [
            "what's model.eval?",
            "why does view() fail after permute()?",
            "why does x.item() fail on tensor with more than one element?",
            "what's the difference between torch.tensor and torch.from_numpy?",
            "is torch.memory_portal a real pytorch api?",
            "what is torch.quantum_backprop?",
            "what is nn.SuperLayer?",
        ]
        model, tokenizer = load_checkpoint_runtime(device=device)
        for q in questions:
            result = run_inference(model=model, tokenizer=tokenizer, query=q)
            print_inference_result(result=result)
