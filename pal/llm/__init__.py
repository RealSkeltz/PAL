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


PROVIDERS = ("ollama", "anthropic")


def get_client(spec: str, **kwargs) -> LLMClient:
    provider, sep, model = spec.partition(":")
    # ollama tags contain colons ("qwen3.5:9b"), so only split off a prefix that
    # is actually a provider name; anything else is a bare ollama model.
    if not sep or provider not in PROVIDERS:
        provider, model = "ollama", spec

    if provider == "ollama":
        from pal.llm.ollama_client import OllamaClient
        return OllamaClient(model, **kwargs)

    if provider == "anthropic":
        from pal.llm.anthropic_client import AnthropicClient
        return AnthropicClient(model, **kwargs)

    raise ValueError(f"Unknown LLM provider '{provider}' in '{spec}'")
