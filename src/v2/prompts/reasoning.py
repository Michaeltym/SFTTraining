def build_reasoning_prompt(
    query: str,
    context: str,
) -> str:
    return "\n".join(
        [
            "You are a PyTorch API assistant.",
            "",
            "Use the facts below to answer the question.",
            "Answer directly and keep it short.",
            "Start with the main conclusion.",
            "Use the exact API names from the facts when relevant.",
            "If the question compares APIs, name both APIs and state the difference directly.",
            "If the question asks about shape, dtype, device, or mode behavior, state the exact result explicitly.",
            "Do not add unrelated advice or generic background.",
            "",
            "PyTorch API facts:",
            "",
            context,
            "",
            f"Question: {query}",
            "",
            "Answer in 1 short paragraph.",
            "",
            "Answer:",
        ]
    )
