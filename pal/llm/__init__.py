"""Pluggable LLM backends.

Models are named `provider:model`, e.g. `ollama:qwen3.5:9b` or
`anthropic:claude-opus-5`. A bare name with no provider is treated as ollama.
"""
from pal.llm.base import (
    Completed,
    LLMClient,
    TextDelta,
    ToolCall,
    ToolSpec,
    Turn,
)

__all__ = [
    "Completed", "LLMClient", "TextDelta", "ToolCall", "ToolSpec", "Turn", "get_client",
]


PROVIDERS = ("ollama", "anthropic", "foundry")


def get_client(spec: str, **kwargs) -> LLMClient:
    """Build a client from a `provider:model` spec.

    A bare provider name ("foundry") uses that backend's default model; anything
    else without a recognised prefix is treated as an ollama model, since ollama
    tags contain colons of their own ("qwen3.5:9b").
    """
    head, sep, tail = spec.partition(":")

    if spec in PROVIDERS:
        provider, model = spec, None          # provider default model
    elif sep and head in PROVIDERS:
        provider, model = head, tail
    else:
        provider, model = "ollama", spec

    if model is not None:
        kwargs["model"] = model

    if provider == "ollama":
        from pal.llm.ollama_client import OllamaClient
        return OllamaClient(**kwargs)

    if provider == "anthropic":
        from pal.llm.anthropic_client import AnthropicClient
        return AnthropicClient(**kwargs)

    from pal.llm.anthropic_client import FoundryClient
    return FoundryClient(**kwargs)
