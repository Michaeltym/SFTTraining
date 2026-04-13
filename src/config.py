from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 100
USE_CHAT_TEMPLATE = False
BATCH_SIZE = 8
IGNORE_INDEX = -100
EPOCHS = 1
LEARNING_RATE = 1e-4
DATASET_NAME = "dataset_3"

HF_CACHE_DIR = Path("./data/hf_home")
EVAL_RESULTS_DIR = Path("./experiments/eval_results")
RAW_DATA_DIR = Path("./data/raw")
CHECKPOINT_DIR = Path("./data/checkpoints")
ADAPTER_DIR = Path("./data/adapters")
CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-{BATCH_SIZE}-{LEARNING_RATE}.pt"
)

MODE_BASELINE = "baseline"
MODE_TRAIN = "train"
MODE_RESUME = "resume"
MODE_EVALUATE = "evaluate"
MODE = MODE_EVALUATE

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
