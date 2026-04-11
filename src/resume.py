import torch
from src.model import load_model
from src.checkpoint import load_checkpoint
from src.training import get_dataloaders, run_training, run_validate
from src.config import CHECKPOINT_PATH, EPOCHS


def run_resume(device: torch.device):
    checkpoint = load_checkpoint(path=CHECKPOINT_PATH, device=device)
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    epoch = checkpoint["epoch"]
    best_validation_loss = checkpoint["validation_loss"]
    batch_size = checkpoint["batch_size"]
    learning_rate = checkpoint["learning_rate"]
    model_name = checkpoint["model_name"]
    dataset_name = checkpoint["dataset_name"]
    print(f"##### Resuming training from checkpoint {CHECKPOINT_PATH} #####")
    model = load_model(model_name=model_name)
    model.load_state_dict(model_state_dict)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(optimizer_state_dict)
    training_dataloader, validation_dataloader = get_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size, model_name=model_name
    )
    for epoch in range(epoch + 1, EPOCHS):
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
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_name=model_name,
            model=model,
            training_loss=training_loss,
            dataset_name=dataset_name,
        )
