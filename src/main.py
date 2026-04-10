import torch
from src.baseline import run_baseline
from src.config import MODE, MODE_BASE, MODE_TRAIN, BATCH_SIZE
from src.data import load_jsonl_data, format_jsonl_data, encode_data
from src.dataset import CustomDataset
from src.dataloader import build_dataloader
from src.train import run_train

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    if MODE == MODE_BASE:
        run_baseline(device=device)
    elif MODE == MODE_TRAIN:
        dataset = load_jsonl_data(file_path="./data/raw/sft_dataset_v1.jsonl")
        formatted_data = format_jsonl_data(dataset=dataset)
        encoded_data = encode_data(formatted_data)
        custom_dataset = CustomDataset(encoded=encoded_data)
        dataloader = build_dataloader(
            dataset=custom_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        run_train(dataloader=dataloader, device=device)
