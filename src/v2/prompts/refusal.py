def build_refusal_prompt(query: str, context: str) -> str:
    return "\n".join(
        [
            "You are a PyTorch API assistant.",
            "",
            "The retrieved PyTorch docs do not confirm this API.",
            "",
            "If the API is not supported by the retrieved PyTorch docs, do not guess.",
            "",
            context,
            "",
            "If the query depends on an unconfirmed symbol, do not infer its behavior.",
            "",
            "Answer in one short paragraph.",
            "",
            "State clearly that this API does not exist in the retrieved PyTorch docs or cannot be verified from them.",
            "",
            f"Question: {query}",
            "",
            "Answer:",
        ]
    )
