from pathlib import Path
from typing import TypedDict
import torch


class Checkpoint(TypedDict):
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, torch.Tensor]
    epoch: int
    validation_loss: float
    batch_size: int
    learning_rate: float
    model_name: str
    training_loss: float
    dataset_name: str


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print("New checkpoint saved!")


def load_checkpoint(path: Path, device: torch.device) -> Checkpoint:
    return torch.load(path, map_location=device, weights_only=False)
