from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 128
RAG_MAX_NEW_TOKENS = 64
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
MODE = MODE_HYBRID_WITH_BASE_MODEL

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
