from transformers import AutoModelForCausalLM
from typing import Any
from peft import LoraConfig, TaskType, get_peft_model
from src.config import (
    HF_CACHE_DIR,
    LORA_TARGET_MODULES,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
)


def load_model(model_name: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)

    return model


def load_model_with_adapter(model_name: str) -> Any:
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=LORA_DROPOUT,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def get_model_name_slug(model_name: str) -> str:
    return model_name.replace("/", "-")
