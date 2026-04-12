from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 100
USE_CHAT_TEMPLATE = False
BATCH_SIZE = 16
IGNORE_INDEX = -100
EPOCHS = 2
LEARNING_RATE = 2e-5
DATASET_NAME = "dataset_3"

HF_CACHE_DIR = Path("./data/hf_home")
EVAL_RESULTS_DIR = Path("./experiments/eval_results")
RAW_DATA_DIR = Path("./data/raw")
CHECKPOINT_DIR = Path("./data/checkpoints")
CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-{BATCH_SIZE}-{LEARNING_RATE}.pt"
)

MODE_BASELINE = "baseline"
MODE_TRAIN = "train"
MODE_RESUME = "resume"
MODE_EVALUATE = "evaluate"
MODE = MODE_TRAIN
