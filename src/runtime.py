import torch
from src.model import load_model
from src.checkpoint import load_checkpoint, Checkpoint
from src.adapter import load_adapter
from src.tokenizer import load_tokenizer
from src.config import CHECKPOINT_PATH


def build_checkpoint_runtime(checkpoint: Checkpoint, device: torch.device):
    model_name = checkpoint["model_name"]
    adapter_path = checkpoint["adapter_path"]
    model = load_model(model_name=model_name)
    model = load_adapter(adapter_path=adapter_path, model=model, is_trainable=False)
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer(model_name=model_name)
    return model, tokenizer


def load_checkpoint_runtime(device: torch.device):
    checkpoint = load_checkpoint(path=CHECKPOINT_PATH, device=device)
    return build_checkpoint_runtime(checkpoint=checkpoint, device=device)
