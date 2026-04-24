from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 128
RAG_MAX_NEW_TOKENS = 64
USE_CHAT_TEMPLATE = False
BATCH_SIZE = 8
IGNORE_INDEX = -100
EPOCHS = 2
LEARNING_RATE = 1e-4
DATASET_NAME = "dataset_10"

# Optional slice for the SFT data loader. Set to a positive integer to cap
# the number of rows used from training/validation jsonl, or set to None to
# use the full dataset. Useful for a pilot run before a full scale-up.
# Reset both to None when running the full dataset.
#
# ds10 is the 200-row / 24-row scale-up per CLAUDE.md. The jsonl files are
# already the final size, so setting caps to None uses the full file.
MAX_TRAIN_ROWS: int | None = None
MAX_VAL_ROWS: int | None = None

# Seed used to shuffle a jsonl dataset before slicing with MAX_TRAIN_ROWS or
# MAX_VAL_ROWS. With a fixed seed the slice is reproducible across runs but
# is no longer biased by the file's original row order. Set to None to skip
# the shuffle (i.e. take the first N rows verbatim, only safe when the file
# is already random or the cap equals the full dataset).
PILOT_SHUFFLE_SEED: int | None = 42

HF_CACHE_DIR = Path("./data/hf_home")
EVAL_RESULTS_DIR = Path("./experiments/eval_results")
RAW_DATA_DIR = Path("./data/raw")
CHECKPOINT_DIR = Path("./data/checkpoints")
ADAPTER_DIR = Path("./data/adapters")
KNOWLEDGE_DIR = Path("./data/knowledge/pytorch_docs")
CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / f"{MODEL_NAME.replace('/', '-')}-{DATASET_NAME}-{BATCH_SIZE}-{LEARNING_RATE}.pt"
)
BENCHMARK_NAME = "core"
BENCHMARK_DATA_PATH = Path(f"./data/eval/benchmark_{BENCHMARK_NAME}_pytorch.jsonl")
BENCHMARK_RESULTS_DIR = Path("./experiments/eval_results/benchmark")
PYTORCH_DOCS_SOURCE_DIR = Path("./data/source/pytorch_docs")
PYTORCH_CORPUS_OUTPUT_PATH = Path("./data/output/cache/pytorch_corpus.jsonl")
PYTORCH_SYMBOL_INDEX_OUTPUT_PATH = Path("./data/output/cache/pytorch_symbol_index.json")

MODE_BASELINE = "baseline"
MODE_TRAIN = "train"
MODE_RESUME = "resume"
MODE_EVALUATE = "evaluate"
MODE_INFERENCE = "inference"
MODE_RAG_EVALUATE = "rag_evaluate"
MODE_HYBRID = "hybrid"
MODE_HYBRID_WITH_BASE_MODEL = "hybrid_with_base_model"
MODE = MODE_HYBRID

RAG_RETRIEVAL_TITLE_TOKEN_WEIGHT = 5
RAG_RETRIEVAL_TAG_TOKEN_WEIGHT = 3
RAG_RETRIEVAL_TEXT_TOKEN_WEIGHT = 1
RAG_RETRIEVAL_TITLE_SYMBOL_BASE = 15
RAG_RETRIEVAL_TITLE_SYMBOL_STEP = 5
RAG_RETRIEVAL_TAG_SYMBOL_BASE = 8
RAG_RETRIEVAL_TAG_SYMBOL_STEP = 4
RAG_RETRIEVAL_TEXT_SYMBOL_BASE = 3
RAG_RETRIEVAL_TEXT_SYMBOL_STEP = 1
RAG_RETRIEVAL_COMPARISON_BONUS = 50

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

HALLUCINATION_REFUSAL_THRESHOLD = 3
