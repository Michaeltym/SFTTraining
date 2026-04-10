from pathlib import Path

PRETRAINED_MODEL = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKEN = 100
USE_CHAT_TEMPLATE = False
BATCH_SIZE = 16
IGNORE_INDEX = -100
EPOCHS = 10
LEARNING_RATE = 2e-5

HF_CACHE_DIR = Path("./data/hf_home")
EVAL_RESULTS_DIR = Path("./experiments/eval_results")
RAW_DATA_DIR = Path("./data/raw")

MODE_BASE = "base"
MODE_TRAIN = "train"
MODE = MODE_TRAIN
