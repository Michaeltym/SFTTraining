from transformers import AutoModelForCausalLM
from typing import Any
from src.config import HF_CACHE_DIR


def load_model(model_name: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    return model


def get_model_name_slug(model_name: str) -> str:
    return model_name.replace("/", "-")
