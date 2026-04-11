import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from typing import Any
from src.model import get_model_name_slug
from src.checkpoint import save_checkpoint
from src.config import CHECKPOINT_DIR


def run_validate(
    best_validation_loss: float,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    epoch: int,
    batch_size: int,
    learning_rate: float,
    model_name: str,
    model: Any,
    training_loss: float,
    dataset_name: str,
) -> float:
    print(f"##### Epoch {epoch + 1} validation started #####")
    model.eval()
    total_steps = len(dataloader)
    running_loss = 0
    start_time = time.time()
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = output.loss
            running_loss += loss.item()
        validation_loss = running_loss / total_steps
        print(f"##### Validation finished #####")
        print(f"Avg loss: {validation_loss:.4f}")
        print(f"Time: {(time.time() - start_time):.2f}s")
        if validation_loss < best_validation_loss:
            save_checkpoint(
                path=CHECKPOINT_DIR
                / f"{get_model_name_slug(model_name)}-{dataset_name}-{batch_size}-{learning_rate}.pt",
                checkpoint={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "model_name": model_name,
                    "training_loss": training_loss,
                    "dataset_name": dataset_name,
                },
            )
            return validation_loss
        return best_validation_loss
