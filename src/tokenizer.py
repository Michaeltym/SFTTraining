from transformers import AutoTokenizer
from src.config import HF_CACHE_DIR


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    return tokenizer
