"""Local models served by ollama.

Ollama unloads a model after 5 idle minutes and the next request then waits
seconds for it to load again, so PAL keeps its models loaded for as long as it
runs (keep_alive=-1) and releases them on exit.
"""
from uuid import uuid4

import ollama

from pal.llm.base import Completed, LLMClient, TextDelta, ToolCall, ToolSpec, Turn


class OllamaClient(LLMClient):
    def __init__(self, model: str, think: bool = False):
        self.model = model
        self.think = think

    def warm_up(self) -> bool:
        ollama.generate(model=self.model, keep_alive=-1)
        return True

    def release(self):
        ollama.generate(model=self.model, keep_alive=0)

    def parse(self, system, conversation, output):
        messages = ([{"role": "system", "content": system}] if system else []) + [_encode_turn(t) for t in conversation]
        response = ollama.chat(model=self.model, messages=messages, format=output.model_json_schema(),
                               think=self.think, keep_alive=-1)
        return output.model_validate_json(response["message"]["content"])

    def stream(self, system, conversation, tools=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(_encode_turn(t) for t in conversation)

        stream = ollama.chat(
            model=self.model,
            messages=messages,
            tools=[_encode_tool(t) for t in (tools or [])],
            think=self.think,
            stream=True,
            keep_alive=-1,
        )

        text = ""
        calls: list[ToolCall] = []
        for chunk in stream:
            message = chunk["message"]

            for call in message.get("tool_calls") or []:
                fn = call["function"]
                # ollama doesn't issue call ids, but we mint one anyway so a
                # conversation stays replayable if the backend is swapped.
                calls.append(ToolCall(
                    name=fn["name"],
                    arguments=fn.get("arguments") or {},
                    id=f"toolu_{uuid4().hex[:20]}",
                ))

            piece = message.get("content") or ""
            if piece:
                text += piece
                yield TextDelta(piece)

        yield Completed(text=text, tool_calls=calls)


def _encode_turn(turn: Turn) -> dict:
    # ollama has no tool_use ids — a tool result is just another message.
    if turn.role == "tool":
        return {"role": "tool", "content": turn.text}

    message = {"role": turn.role, "content": turn.text}
    if turn.images:
        message["images"] = turn.images
    if turn.tool_calls:
        message["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in turn.tool_calls
        ]
    return message


def _encode_tool(spec: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": spec.properties,
                "required": spec.required,
            },
        },
    }
