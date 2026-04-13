import torch
from typing import Any, TypedDict
from src.rag.knowledge import load_knowledge
from src.rag.retriever import retrieve_top_k_knowledge
from src.rag.prompt import build_prompt
from src.config import MAX_NEW_TOKENS


class InferenceSource(TypedDict):
    id: str
    title: str
    url: str
    weight: float


class InferenceResult(TypedDict):
    query: str
    answer: str
    sources: list[InferenceSource]


def run_inference(model: Any, tokenizer: Any, query: str) -> InferenceResult:
    knowledge_items = load_knowledge()
    retrieved_knowledge = retrieve_top_k_knowledge(
        knowledge_items=knowledge_items,
        query=query,
        top_k=1,
    )
    prompt = build_prompt(query=query, retrieved_knowledge=retrieved_knowledge)
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        sources: list[InferenceSource] = [
            {
                "id": item["knowledge"]["id"],
                "title": item["knowledge"]["title"],
                "url": item["knowledge"]["url"],
                "weight": item["weight"],
            }
            for item in retrieved_knowledge
        ]
        return {"query": query, "answer": output_text, "sources": sources}


def print_inference_result(result: InferenceResult):
    sources_block = []
    for i, source in enumerate(result["sources"], start=1):
        sources_block.append(
            "\n".join(
                [
                    f"Source {i}",
                    f"ID: {source['id']}",
                    f"Title: {source['title']}",
                    f"URL: {source['url']}",
                    f"Weight: {source['weight']}",
                ]
            )
        )
    sources = "\n\n".join(sources_block)
    output = "\n".join(
        [
            "Question:",
            result["query"],
            "",
            "Answer:",
            result["answer"],
            "",
            "Sources:",
            "",
            sources,
        ]
    )
    print(output)
    print("\n" + "=" * 80 + "\n")
