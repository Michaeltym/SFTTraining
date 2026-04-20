from typing import TypedDict


class BuildPromptResult(TypedDict):
    prompt: str
    should_use_refusal: bool
