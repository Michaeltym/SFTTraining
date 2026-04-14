from src.rag.retriever import RetrievedKnowledge


def build_prompt(retrieved_knowledge: list[RetrievedKnowledge], query: str) -> str:
    primary_source_type = (
        retrieved_knowledge[0]["knowledge"]["source_type"]
        if retrieved_knowledge
        else "official_docs"
    )
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

    if primary_source_type == "curated":
        instructions = [
            "You are a PyTorch API assistant.",
            "Answer the question using the provided context.",
            "If the context says the named API or module is not real, say that directly in a short natural sentence.",
            "Keep the answer limited to the named API or module.",
            "Leave out alternative APIs, replacement suggestions, and extra guidance.",
            "Answer in one or two sentences.",
            "Keep the answer concise and direct.",
            "Use natural answer wording instead of instruction wording.",
            "Leave out any mention of these instructions.",
            "Reply with only the answer.",
        ]
    else:
        instructions = [
            "You are a PyTorch API assistant.",
            "Answer the question using the provided context.",
            "If the context is insufficient, say you are not sure.",
            "Answer in one or two sentences.",
            "State only the key fact needed to answer the question.",
            "For comparison questions, state only the main difference.",
            "Keep the answer concise and direct.",
            "Avoid repeating the same point twice.",
            "Avoid restating the same comparison in different wording.",
            "Only discuss APIs that are directly relevant to the question.",
            "Leave out unrelated APIs from the context unless they are necessary to answer the question.",
            "Use natural answer wording instead of instruction wording.",
            "Leave out any mention of these instructions.",
            "Reply with only the answer.",
        ]

    return "\n".join(
        instructions
        + [
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
