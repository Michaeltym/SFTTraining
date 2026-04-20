def build_reasoning_prompt(
    query: str,
    context: str,
    unconfirmed_symbols: list[str],
) -> str:
    lines = [
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
    ]
    if unconfirmed_symbols:
        sym_str = ", ".join(f"`{s}`" for s in unconfirmed_symbols)
        lines.append(
            f"IMPORTANT: {sym_str} is not confirmed in the retrieved PyTorch docs. "
            f"You must state it is not a valid PyTorch API, "
            f"and you must not describe its behavior."
        )
        lines.append("")
    lines.extend(
        [
            f"Question: {query}",
            "",
            "Answer in 1 short paragraph.",
            "",
            "Answer:",
        ]
    )
    return "\n".join(lines)
