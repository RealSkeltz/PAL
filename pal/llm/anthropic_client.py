"""Claude, either on the first-party API or through Microsoft Foundry.

`anthropic` is imported lazily so the ollama path never depends on it.
"""
import logging
import os

from pal.llm.base import Completed, LLMClient, TextDelta, ToolCall, ToolSpec, Turn

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
FOUNDRY_DEFAULT_MODEL = "claude-sonnet-4-6"

# Spoken replies are short by construction — a few sentences at most. A tight
# budget keeps a rambling turn from stalling the voice loop.
DEFAULT_MAX_TOKENS = 2048

# Effort is the supported lever for a latency-sensitive loop. On Opus 5 thinking
# is on by default and disabling it is a known trap (tool calls can arrive as
# plain text and silently never run), so we leave `thinking` unset and lean on
# low effort instead. Note the default differs by model: omitting `thinking`
# means adaptive on Opus 5 but *no* thinking on Sonnet 4.6 — both acceptable
# here, since the fastest turnaround is what a voice loop wants.
DEFAULT_EFFORT = "low"

# Opt-in server-side retry: if a request is declined on policy grounds, the API
# re-runs it on a substitute model in the same call rather than returning empty.
# First-party API only — not available on Foundry, Bedrock, or Vertex.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Prompt caching: every reply resends the same system prompt and a conversation that only
# grows at the end, so most of the input can be read from cache instead of processed again.
# One breakpoint after the fixed system prompt (it survives memory changing the rest), plus
# automatic caching of the conversation tail. PAL_PROMPT_CACHE=0 turns it off, for comparing.
PROMPT_CACHE = os.environ.get("PAL_PROMPT_CACHE", "1") != "0"


class AnthropicClient(LLMClient):
    """Claude on the first-party API."""

    supports_fallbacks = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = self._connect()

    def _connect(self):
        # Credentials resolve from ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile — nothing to pass here.
        return _sdk().Anthropic()

    def parse(self, system, conversation, output):
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": _encode_conversation(conversation),
            "output_config": {"effort": self.effort},
            "output_format": output,  # the SDK merges this into output_config as a JSON schema
        }
        if system:
            request["system"] = system
        response = self._client.messages.parse(**request)
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{self.model} declined to answer")
        return response.parsed_output

    def stream(self, system, conversation, tools=None):
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": _encode_conversation(conversation),
            "output_config": {"effort": self.effort},
        }
        if self.supports_fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        parts = [system] if isinstance(system, str) else [p for p in (system or []) if p]
        if parts:
            request["system"] = [{"type": "text", "text": p} for p in parts]
            if PROMPT_CACHE:
                request["system"][0]["cache_control"] = {"type": "ephemeral"}
        if PROMPT_CACHE:
            request["cache_control"] = {"type": "ephemeral"}
        if tools:
            request["tools"] = [_encode_tool(t) for t in tools]

        with self._client.beta.messages.stream(**request) as stream:
            for piece in stream.text_stream:
                yield TextDelta(piece)
            final = stream.get_final_message()
        u = final.usage
        logger.debug(f"usage: input {u.input_tokens} + cache read {u.cache_read_input_tokens or 0} "
                     f"+ cache write {u.cache_creation_input_tokens or 0}, output {u.output_tokens}")

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


class FoundryClient(AnthropicClient):
    """Claude hosted on Microsoft Foundry.

    Same wire format as the first-party API, but a different client class,
    different credentials, and no server-side refusal fallbacks.
    """

    # `fallbacks` is Claude-API-only; sending it here is rejected.
    supports_fallbacks = False

    def __init__(self, model: str = FOUNDRY_DEFAULT_MODEL, **kwargs):
        super().__init__(model, **kwargs)

    def _connect(self):
        # Reads ANTHROPIC_FOUNDRY_API_KEY / _RESOURCE / _BASE_URL from the
        # environment — the first-party ANTHROPIC_API_KEY is not consulted,
        # which is the easy mistake to make here.
        try:
            return _sdk().AnthropicFoundry()
        except Exception as e:
            raise RuntimeError(
                "Foundry needs ANTHROPIC_FOUNDRY_API_KEY plus one of "
                "ANTHROPIC_FOUNDRY_RESOURCE or ANTHROPIC_FOUNDRY_BASE_URL. "
                "ANTHROPIC_API_KEY is a different credential and is not used here. "
                f"({e})"
            ) from e


def _sdk():
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The Anthropic backend needs the anthropic SDK: poetry add anthropic"
        ) from e
    return anthropic


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
