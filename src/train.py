import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from src.model import load_model
from src.config import EPOCHS, LEARNING_RATE


def run_train(dataloader: DataLoader, device: torch.device):
    model = load_model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(dataloader)
    model.train()
    print("##### Training started #####")
    for epoch in range(EPOCHS):
        running_loss = 0
        start_time = time.time()
        print(f"##### Training epoch {epoch + 1} started #####")
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

            if (step + 1) % 10 == 0:
                print(
                    f"Epoch {epoch} step {step + 1}/{total_steps} "
                    f"Avg loss: {(running_loss / (step + 1)):.4f} Time: {(time.time() - start_time):.2f}s"
                )

        average_training_loss = running_loss / total_steps
        print(f"##### Training epoch {epoch + 1} finished #####")
        print(f"Avg loss: {average_training_loss:.4f}")
        print(f"Time: {(time.time() - start_time):.2f}s")
