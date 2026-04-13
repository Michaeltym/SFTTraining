from src.rag.retriever import RetrievedKnowledge


def build_prompt(retrieved_knowledge: list[RetrievedKnowledge], query: str) -> str:
    context_blocks = []

    for i, item in enumerate(retrieved_knowledge, start=1):
        knowledge = item["knowledge"]
        context_blocks.append(
            "\n".join(
                [
                    f"Context {i}:",
                    f"Title: {knowledge['title']}",
                    f"URL: {knowledge['url']}",
                    f"Source Type: {knowledge['source_type']}",
                    f"Tags: {', '.join(knowledge['tags'])}",
                    f"Text: {knowledge['text']}",
                ]
            )
        )

    contexts = (
        "\n\n".join(context_blocks)
        if context_blocks
        else "No supporting context found."
    )

    return "\n".join(
        [
            "You are a PyTorch API assistant.",
            "Answer the question using the provided context.",
            "If the context is insufficient, say you are not sure.",
            "Do not invent nonexistent PyTorch APIs.",
            "Answer in one or two sentences.",
            "State only the key fact needed to answer the question.",
            "Keep the answer concise and direct.",
            "Do not repeat the same point twice.",
            "Only discuss APIs that are directly relevant to the question.",
            "Do not add unrelated APIs from the context unless they are necessary to answer the question.",
            "Do not repeat or quote the instructions.",
            "Reply with only the final answer.",
            "",
            f"Question: {query}",
            "",
            "Context:",
            "",
            contexts,
            "",
            "Answer:",
        ]
    )
