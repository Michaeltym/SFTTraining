import torch
from src.baseline import run_baseline
from src.training import run_training_loop
from src.resume import run_resume
from src.evaluate import run_evaluate
from src.config import MODE, MODE_BASELINE, MODE_RESUME, MODE_TRAIN, MODE_EVALUATE


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
