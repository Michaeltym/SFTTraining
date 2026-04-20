def build_refusal_prompt(
    query: str,
    context: str,
    unconfirmed_symbols: list[str],
) -> str:
    lines = [
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
            "Answer in one short paragraph.",
            "",
            "State clearly that this API does not exist in the retrieved PyTorch docs or cannot be verified from them.",
            "",
            f"Question: {query}",
            "",
            "Answer:",
        ]
    )
    return "\n".join(lines)
