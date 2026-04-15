import torch
from src.v2.benchmark.run import run_benchmark
from src.model import load_model
from src.tokenizer import load_tokenizer
from src.config import (
    MODEL_NAME,
    MODE,
    MODE_BASELINE,
)

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    device = torch.device(device)
    if MODE == MODE_BASELINE:
        model = load_model(model_name=MODEL_NAME)
        model.to(device)
        tokenizer = load_tokenizer(model_name=MODEL_NAME)
        run_benchmark(
            model=model,
            tokenizer=tokenizer,
            system_name="baseline_raw_model",
            model_name=MODEL_NAME,
            mode=MODE_BASELINE,
        )
