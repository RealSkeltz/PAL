"""Claude via the Anthropic API.

`anthropic` is imported lazily so the ollama path never depends on it.
"""
from pal.llm.base import Completed, LLMClient, TextDelta, ToolCall, ToolSpec, Turn

DEFAULT_MODEL = "claude-opus-5"

# Spoken replies are short by construction — a few sentences at most. A tight
# budget keeps a rambling turn from stalling the voice loop.
DEFAULT_MAX_TOKENS = 2048

# Thinking is on by default on Opus 5 and disabling it is a known trap (tool
# calls can arrive as plain text and silently never run). Low effort is the
# supported way to keep a latency-sensitive loop cheap and fast.
DEFAULT_EFFORT = "low"

# Opt-in server-side retry: if a request is declined on policy grounds, the API
# re-runs it on a substitute model in the same call rather than returning empty.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The Anthropic backend needs the anthropic SDK: poetry add anthropic"
            ) from e

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        # Credentials resolve from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile — nothing to pass here.
        self._client = anthropic.Anthropic()

    def stream(self, system, conversation, tools=None):
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": _encode_conversation(conversation),
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [_encode_tool(t) for t in tools]

        with self._client.beta.messages.stream(**request) as stream:
            for piece in stream.text_stream:
                yield TextDelta(piece)
            final = stream.get_final_message()

        # Check stop_reason before reading content — a refusal can leave it empty.
        if final.stop_reason == "refusal":
            yield Completed(refused=True)
            return

        text = "".join(b.text for b in final.content if b.type == "text")
        calls = [
            ToolCall(name=b.name, arguments=b.input or {}, id=b.id)
            for b in final.content
            if b.type == "tool_use"
        ]
        yield Completed(text=text, tool_calls=calls)


def _encode_conversation(turns: list[Turn]) -> list[dict]:
    """Turns → Anthropic messages.

    Tool results are user-role content blocks, and every result for one assistant
    turn must arrive in a single message — splitting them trains Claude out of
    parallel tool calls. Consecutive tool turns are therefore batched.
    """
    messages: list[dict] = []
    pending_results: list[dict] = []

    def flush_results():
        if pending_results:
            messages.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for turn in turns:
        if turn.role == "tool":
            pending_results.append({
                "type": "tool_result",
                "tool_use_id": turn.tool_call_id,
                "content": turn.text,
            })
            continue

        flush_results()
        blocks = []

        if turn.role == "user":
            # Image first, then the text that refers to it.
            blocks.extend(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": img},
                }
                for img in turn.images
            )

        if turn.text:
            blocks.append({"type": "text", "text": turn.text})

        blocks.extend(
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in turn.tool_calls
        )

        if blocks:  # the API rejects a message with empty content
            messages.append({"role": turn.role, "content": blocks})

    flush_results()
    return messages


def _encode_tool(spec: ToolSpec) -> dict:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": {
            "type": "object",
            "properties": spec.properties,
            "required": spec.required,
        },
    }
