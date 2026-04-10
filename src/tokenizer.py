from transformers import AutoTokenizer
from src.config import PRETRAINED_MODEL, HF_CACHE_DIR


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, cache_dir=HF_CACHE_DIR)
    return tokenizer
