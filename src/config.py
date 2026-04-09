from pathlib import Path

PRETRAINED_MODEL = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKEN = 100
USE_CHAT_TEMPLATE = False

HF_CACHE_DIR = Path("./data/hf_home")
EVAL_RESULTS_DIR = Path("./experiments/eval_results")

MODE_BASE = "base"
MODE_TRAINED = "trained"
MODE = MODE_BASE
