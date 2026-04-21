from datasets import load_dataset, Dataset
from typing import TypedDict
import torch
from src.tokenizer import load_tokenizer
from src.config import IGNORE_INDEX


class EncodedData(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def load_jsonl_data(
    file_path: str,
    max_rows: int | None = None,
    shuffle_seed: int | None = None,
) -> Dataset:
    dataset = load_dataset("json", data_files=file_path, split="train")
    if shuffle_seed is not None and max_rows is not None and max_rows < len(dataset):
        dataset = dataset.shuffle(seed=shuffle_seed)
    if max_rows is not None and max_rows < len(dataset):
        dataset = dataset.select(range(max_rows))
    return dataset


def format_jsonl_data(dataset: Dataset) -> list[dict[str, str]]:
    formatted_data: list[dict[str, str]] = []
    for datum in dataset:
        input_text = datum["input"].strip()
        output_text = datum["output"]
        prompt_parts: list[str] = []
        if input_text:
            prompt_parts.append(f"Input:\n{input_text}\n")
        prompt_parts.append("Response:\n")
        prompt_text = "".join(prompt_parts)
        formatted_data.append(
            {
                "prompt_text": prompt_text,
                "full_text": f"{prompt_text}{output_text}",
            }
        )
    return formatted_data


def encode_data(
    formatted_data: list[dict[str, str]],
    model_name: str,
) -> list[EncodedData]:
    tokenizer = load_tokenizer(model_name=model_name)
    encoded_data = []
    for datum in formatted_data:
        prompt_text = datum["prompt_text"]
        full_text = datum["full_text"]
        encoded_full_text = tokenizer(full_text, truncation=False, return_tensors="pt")
        encoded_prompt_text = tokenizer(
            prompt_text, truncation=False, return_tensors="pt"
        )
        full_text_input_ids = encoded_full_text["input_ids"][-1]
        labels = full_text_input_ids.clone()
        prompt_text_length = encoded_prompt_text["input_ids"].shape[-1]
        mask = torch.ones_like(full_text_input_ids)
        mask[:prompt_text_length] = 0
        masked_labels = labels.masked_fill(mask == 0, IGNORE_INDEX)
        encoded_data.append(
            {
                "input_ids": full_text_input_ids,
                "attention_mask": encoded_full_text["attention_mask"][-1],
                "labels": masked_labels,
            }
        )
    return encoded_data
