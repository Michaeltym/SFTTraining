from typing import Any
from peft import PeftModel
from src.config import ADAPTER_DIR


def save_adapter(filename: str, model: Any) -> None:
    adapter_path = ADAPTER_DIR / filename
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)


def load_adapter(adapter_path: str, model: Any, is_trainable: bool = False) -> Any:
    model = PeftModel.from_pretrained(
        model=model, model_id=adapter_path, is_trainable=is_trainable
    )
    return model
