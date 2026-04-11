from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.tokenizer import load_tokenizer
from src.dataset import CustomDataset
from src.data import EncodedData
from src.config import IGNORE_INDEX


def collate_fn_factory(tokenizer):
    def collate_fn(batch: list[EncodedData]):
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        padded_input_ids = pad_sequence(
            sequences=input_ids, batch_first=True, padding_value=pad_token_id
        )
        padded_attention_mask = pad_sequence(
            sequences=attention_mask, batch_first=True, padding_value=0
        )
        padded_labels = pad_sequence(
            sequences=labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }

    return collate_fn


def build_dataloader(
    dataset: CustomDataset, batch_size: int, shuffle: bool, model_name: str
) -> DataLoader:
    tokenizer = load_tokenizer(model_name=model_name)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_factory(tokenizer),
    )
    return dataloader
