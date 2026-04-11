import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from typing import Any
from src.data import load_jsonl_data, format_jsonl_data, encode_data
from src.dataset import CustomDataset
from src.dataloader import build_dataloader
from src.validate import run_validate
from src.model import load_model
from src.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    MODEL_NAME,
    DATASET_NAME,
)


def get_dataloaders(
    dataset_name: str, batch_size: int, model_name: str
) -> tuple[DataLoader, DataLoader]:
    training_dataset = load_jsonl_data(
        file_path=f"./data/raw/training/{dataset_name}.jsonl"
    )
    formatted_training_data = format_jsonl_data(dataset=training_dataset)
    encoded_training_data = encode_data(formatted_training_data, model_name=model_name)
    training_dataset = CustomDataset(encoded=encoded_training_data)
    training_dataloader = build_dataloader(
        dataset=training_dataset,
        batch_size=batch_size,
        shuffle=True,
        model_name=model_name,
    )
    validation_dataset = load_jsonl_data(
        file_path=f"./data/raw/validation/{dataset_name}.jsonl"
    )
    formatted_validation_data = format_jsonl_data(dataset=validation_dataset)
    encoded_validation_data = encode_data(
        formatted_validation_data, model_name=model_name
    )
    validation_dataset = CustomDataset(encoded=encoded_validation_data)
    validation_dataloader = build_dataloader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        model_name=model_name,
    )
    return training_dataloader, validation_dataloader


def run_training(
    dataloader: DataLoader,
    device: torch.device,
    model: Any,
    optimizer: optim.Optimizer,
    epoch: int,
):
    print(f"##### Epoch {epoch + 1} training started #####")
    model.train()
    total_steps = len(dataloader)
    running_loss = 0
    start_time = time.time()
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
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
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1} step {step + 1}/{total_steps} "
            f"Avg loss: {(running_loss / (step + 1)):.4f} Time: {(time.time() - start_time):.2f}s"
        )

    average_training_loss = running_loss / total_steps
    print(f"##### Training epoch {epoch + 1} finished #####")
    print(f"Avg loss: {average_training_loss:.4f}")
    print(f"Time: {(time.time() - start_time):.2f}s")

    return average_training_loss


def run_training_loop(device: torch.device):
    training_dataloader, validation_dataloader = get_dataloaders(
        dataset_name=DATASET_NAME, batch_size=BATCH_SIZE, model_name=MODEL_NAME
    )
    best_validation_loss = float("inf")
    model = load_model(model_name=MODEL_NAME)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        training_loss = run_training(
            dataloader=training_dataloader,
            device=device,
            optimizer=optimizer,
            model=model,
            epoch=epoch,
        )
        best_validation_loss = run_validate(
            dataloader=validation_dataloader,
            device=device,
            best_validation_loss=best_validation_loss,
            optimizer=optimizer,
            epoch=epoch,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            model_name=MODEL_NAME,
            model=model,
            training_loss=training_loss,
            dataset_name=DATASET_NAME,
        )
