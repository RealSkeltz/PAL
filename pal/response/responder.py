"""Answer user turns: run the agent, speak its reply, record the result.

Speech is the reply itself, so `respond` drives the voice directly: it needs
to know what was heard to mark each PalTurn done or cancelled. Everything
else is a sink, a callable that takes an AgentEvent and ignores the kinds it
doesn't care about, so adding or swapping an output never touches the agent.
"""
import re
import threading
import time
import traceback
from typing import Callable
from uuid import uuid4

from pal.response.run_turn import AgentEvent, StepDone, TextChunk, ToolFinished, ToolStarted, run_turn
from pal.llm.base import LLMClient
from pal.debug.preview import preview
from pal.log import logger
from pal.schemas.event import Actor, Role, Spoken
from pal.schemas.turn import ConversationTurn, Delivery, PalTurn, Response, UserTurn
from pal.tools.base import ToolHandler
from pal.voice.sounds import tool_tone
from pal.voice.voice import Voice

Sink = Callable[[AgentEvent], None]


def respond(
    history: list[ConversationTurn],
    user_turns: list[UserTurn],
    system_prompt: str | None,
    client: LLMClient,
    voice: Voice,
    tools: ToolHandler | None = None,
    cancel: threading.Event | None = None,
    sinks: list[Sink] | None = None,
):
    cancel = cancel or threading.Event()
    sinks = default_sinks(voice) if sinks is None else sinks
    reply_id = uuid4()             # voice tags are (reply_id, step), unique per reply
    sentences = SentenceBuffer()
    recorded: list[tuple[int, PalTurn]] = []
    step, partial = 0, ""          # partial: this step's text so far, for a cut-off generation

    def say(text):
        logger.info(f"[PAL]      {text}")
        preview.add_event("scout_speech", {"text": text})
        voice.say(text, tag=(reply_id, step))

    for turn in user_turns:
        turn.response = Response.RESPONDING

    status = Response.RESPONDED
    try:
        for event in run_turn(history, system_prompt, client, tools, cancel):
            if isinstance(event, TextChunk):
                partial += event.text
                for sentence in sentences.feed(event.text):
                    say(sentence)
            for sink in sinks:
                sink(event)
            if isinstance(event, StepDone):
                for sentence in sentences.flush():
                    say(sentence)
                if event.text or event.actions:
                    turn = PalTurn(
                        in_reply_to=[t.id for t in user_turns],
                        utterance=Spoken(actor=Actor.PAL, role=Role.DIRECT, text=event.text) if event.text else None,
                        actions=event.actions,
                    )
                    history.append(turn)
                    recorded.append((step, turn))
                step, partial = step + 1, ""
        voice.wait(cancel)
    except Exception:
        logger.error("turn failed:\n" + traceback.format_exc())
        status = Response.FAILED
        voice.wait(cancel)

    if cancel.is_set():
        voice.stop()
        status = Response.CANCELLED
        if partial.strip():  # cut off while the model was still writing this step
            turn = PalTurn(in_reply_to=[t.id for t in user_turns],
                           utterance=Spoken(actor=Actor.PAL, role=Role.DIRECT, text=partial.strip()))
            history.append(turn)
            recorded.append((step, turn))

    for step_, turn in recorded:
        words = len(turn.utterance.text.split()) if turn.utterance else 0
        heard = voice.heard((reply_id, step_))
        if cancel.is_set() and turn.utterance and (heard < words or step_ == step):
            turn.cancel(at_word=heard)
        else:
            turn.delivery = Delivery.DONE

    for turn in user_turns:
        turn.response = status


class SentenceBuffer:
    """Collects streamed text and hands back whole sentences."""

    def __init__(self):
        self.text = ""

    def feed(self, text: str) -> list[str]:
        self.text += text
        # Cut after the last sentence end that is followed by a space (or is the end so far).
        cut = max((m.end() for m in re.finditer(r"[.!?\n]+(\s|$)", self.text)), default=0)
        done, self.text = self.text[:cut], self.text[cut:]
        return _sentences(done)

    def flush(self) -> list[str]:
        done, self.text = self.text, ""
        return _sentences(done)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


# --- Sinks -------------------------------------------------------------------

def default_sinks(voice: Voice) -> list[Sink]:
    return [tool_sound(voice), log_tools(), preview_tools]


def tool_sound(voice: Voice) -> Sink:
    def sink(event: AgentEvent):
        if isinstance(event, ToolStarted):
            voice.play(tool_tone())
    return sink


def log_tools() -> Sink:
    started = {}

    def sink(event: AgentEvent):
        if isinstance(event, ToolStarted):
            started[event.action.id] = time.time()
            logger.info(f"[Tool]     {event.action.tool}({event.action.args})")
        elif isinstance(event, ToolFinished):
            elapsed_ms = (time.time() - started.pop(event.action.id)) * 1000
            logger.info(f"[Tool]     -> {event.action.result[:100]} ({elapsed_ms:.0f}ms)")
        elif isinstance(event, StepDone) and event.refused:
            logger.info("[PAL]      declined the request")

    return sink


def preview_tools(event: AgentEvent):
    if isinstance(event, ToolStarted):
        preview.add_event("tool_call", {"name": event.action.tool, "args": event.action.args})
    elif isinstance(event, ToolFinished):
        preview.add_event("tool_result", {"name": event.action.tool, "result": event.action.result[:300]})
