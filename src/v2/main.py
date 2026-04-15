import torch
from src.v2.benchmark.run import run_benchmark
from src.v2.answer_fns import (
    build_plain_generation_answer_fn,
    build_rag_answer_fn,
    build_hybrid_answer_fn,
)
from src.model import load_model
from src.tokenizer import load_tokenizer
from src.runtime import build_checkpoint_runtime
from src.checkpoint import load_checkpoint
from src.config import (
    MODEL_NAME,
    MODE,
    MODE_BASELINE,
    MODE_EVALUATE,
    MODE_RAG_EVALUATE,
    MODE_HYBRID,
    CHECKPOINT_PATH,
)

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    device = torch.device(device)
    if MODE == MODE_BASELINE:
        model = load_model(model_name=MODEL_NAME)
        model.to(device)
        tokenizer = load_tokenizer(model_name=MODEL_NAME)
        answer_fn = build_plain_generation_answer_fn(model=model, tokenizer=tokenizer)

        run_benchmark(
            system_name="baseline_raw_model",
            model_name=MODEL_NAME,
            mode=MODE_BASELINE,
            answer_fn=answer_fn,
        )
    if MODE == MODE_EVALUATE:
        checkpoint = load_checkpoint(path=CHECKPOINT_PATH, device=device)
        model_name = checkpoint["model_name"]
        model, tokenizer = build_checkpoint_runtime(
            checkpoint=checkpoint, device=device
        )
        answer_fn = build_plain_generation_answer_fn(model=model, tokenizer=tokenizer)

        run_benchmark(
            system_name="post_sft",
            model_name=model_name,
            mode=MODE_EVALUATE,
            answer_fn=answer_fn,
        )
    if MODE == MODE_RAG_EVALUATE:
        checkpoint = load_checkpoint(path=CHECKPOINT_PATH, device=device)
        model_name = checkpoint["model_name"]
        model, tokenizer = build_checkpoint_runtime(
            checkpoint=checkpoint, device=device
        )
        answer_fn = build_rag_answer_fn(model=model, tokenizer=tokenizer)

        run_benchmark(
            system_name="rag",
            model_name=model_name,
            mode=MODE_RAG_EVALUATE,
            answer_fn=answer_fn,
        )
    if MODE == MODE_HYBRID:
        checkpoint = load_checkpoint(path=CHECKPOINT_PATH, device=device)
        model_name = checkpoint["model_name"]
        model, tokenizer = build_checkpoint_runtime(
            checkpoint=checkpoint, device=device
        )
        answer_fn = build_hybrid_answer_fn(model=model, tokenizer=tokenizer)
        run_benchmark(
            system_name="hybrid",
            model_name=model_name,
            mode=MODE_HYBRID,
            answer_fn=answer_fn,
        )
