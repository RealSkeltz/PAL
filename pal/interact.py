import re

from pal.llm.base import Completed, TextDelta, Turn
from pal.voice.speak import voice_output, play_sound
from pal.utils.timing import timed
from pal.status import State, status
from pal.voice.sounds import tool_tone
from pal.debug.preview import preview


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation, keeping the punctuation."""
    parts = re.findall(r"[^.!?\n]+[.!?\n]+|[^.!?\n]+$", text)
    return [p.strip() for p in parts if p.strip()]


def _speak(text: str):
    print(text)
    status.set_state(State.SPEAKING)
    try:
        voice_output(text)
    finally:
        # Back to thinking, not idle — the turn may still have more to say or a
        # tool to run, and the agent clears the state when it finishes.
        status.set_state(State.THINKING)
    preview.add_event("scout_speech", {"text": text})


def run(conversation, system_prompt=None, client=None, tool_handler=None, max_iterations=5):
    """
    Run an agentic turn. The model may call tools across multiple iterations,
    interleaving speech and actions.

    Mutates `conversation` (a list of pal.llm.Turn) in place to record assistant
    turns and tool results. Speaks each sentence as soon as it completes.
    """
    tool_specs = tool_handler.specs if tool_handler else []

    for iteration in range(max_iterations):
        speech_buffer = ""    # accumulates until a sentence boundary
        result = Completed()

        for event in client.stream(system_prompt, conversation, tool_specs):
            if isinstance(event, Completed):
                result = event
                continue

            speech_buffer += event.text
            # Flush completed sentences to TTS as they land.
            if speech_buffer.endswith((".", "!", "?", "\n")):
                for sentence in _split_sentences(speech_buffer):
                    _speak(sentence)
                speech_buffer = ""

        if result.refused:
            print("[Interact] Model declined the request")
            conversation.append(Turn(role="assistant", text=""))
            return

        # Flush whatever didn't end on punctuation (no tools coming).
        if not result.tool_calls and speech_buffer.strip():
            _speak(speech_buffer.strip())

        conversation.append(
            Turn(role="assistant", text=result.text, tool_calls=result.tool_calls)
        )

        # No tools? We're done.
        if not result.tool_calls:
            return

        play_sound(tool_tone())
        for call in result.tool_calls:
            print(f"[Tools] Calling {call.name}({call.arguments})")

            status.tool_fired(call.name)
            with timed(f"tool_{call.name}"):
                if tool_handler is None:
                    output = "Error: no tool handler configured"
                else:
                    preview.add_event("tool_call", {"name": call.name, "args": call.arguments})
                    output = tool_handler.dispatch(call.name, call.arguments)
                    preview.add_event("tool_result", {"name": call.name, "result": output[:300]})

            print(f"[Tools] Result: {output[:100]}{'...' if len(output) > 100 else ''}")
            conversation.append(Turn(role="tool", text=output, tool_call_id=call.id))

    print(f"[Interact] Hit max_iterations ({max_iterations})")
