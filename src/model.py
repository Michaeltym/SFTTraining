from transformers import AutoModelForCausalLM
from src.config import PRETRAINED_MODEL, HF_CACHE_DIR


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL, cache_dir=HF_CACHE_DIR
    )
    return model
