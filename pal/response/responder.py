"""Answer user turns: run the agent, speak its reply, record the result.

Speech is the reply itself, so `respond` drives the voice directly: it needs
to know what was heard to mark each PalTurn done or cancelled. Everything
else is a sink, a callable that takes an AgentEvent and ignores the kinds it
doesn't care about, so adding or swapping an output never touches the agent.
"""
import logging
import re
import threading
import time
from typing import Callable
from uuid import uuid4

from pal.debug import trace
from pal.debug.preview import preview
from pal.response.run_turn import AgentEvent, StepDone, TextChunk, ToolFinished, ToolStarted, run_turn
from pal.llm.base import LLMClient
from pal.log import record
from pal.schemas.event import Actor, Role, Spoken
from pal.schemas.turn import ConversationTurn, Delivery, PalTurn, Response, UserTurn
from pal.tools.base import ToolHandler
from pal.voice.sounds import tool_tone
from pal.voice.voice import Voice

logger = logging.getLogger(__name__)

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
    hold: threading.Event | None = None,
    discard: threading.Event | None = None,
) -> bool:
    """Answer `user_turns`. With `hold`, the reply was started before the gate decided whether
    to look: nothing is said and no tool runs until `hold` is set. If `discard` is set instead
    (the gate wants to look, and `cancel` is set with it), the reply is dropped without a trace
    and the user turns are pending again, to be answered with the frames. True if discarded."""
    cancel = cancel or threading.Event()
    sinks = default_sinks(voice) if sinks is None else sinks
    reply_id = uuid4()             # voice tags are (reply_id, step), unique per reply
    sentences = SentenceBuffer()
    recorded: list[tuple[int, PalTurn]] = []
    step, partial = 0, ""          # partial: this step's text so far, for a cut-off generation

    start = time.time()
    first_text = first_sentence = None
    held: list[tuple[str, int]] = []  # sentences written before the gate decided, with their step

    def speak(text, at_step):
        nonlocal first_sentence
        first_sentence = first_sentence or time.time()
        logger.info(f"said: {text}", extra={"speech": ("pal", text)})
        voice.say(text, tag=(reply_id, at_step))

    def say(text):
        if cancel.is_set():
            return  # cut off; don't start a new sentence after the voice was stopped
        if hold is not None and not hold.is_set():
            held.append((text, step))
            return
        speak(text, step)

    def wait_for_gate():
        """Block until the gate has decided, then say what was held back if it may be said."""
        while hold is not None and not (hold.is_set() or cancel.is_set()):
            time.sleep(0.01)
        while held and not cancel.is_set():
            speak(*held.pop(0))

    for turn in user_turns:
        turn.response = Response.RESPONDING

    status = Response.RESPONDED
    try:
        for event in run_turn(history, system_prompt, client, tools, cancel):
            if held and hold.is_set():
                wait_for_gate()
            if isinstance(event, ToolStarted) and hold is not None:
                wait_for_gate()  # a tool mustn't run for a reply that may be dropped
                if cancel.is_set():
                    break
            if isinstance(event, TextChunk):
                first_text = first_text or time.time()
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
                    _insert_reply(history, turn, user_turns)
                    recorded.append((step, turn))
                step, partial = step + 1, ""
        wait_for_gate()
        voice.wait(cancel)
    except Exception:
        logger.exception("turn failed")
        status = Response.FAILED
        voice.wait(cancel)

    if discard is not None and discard.is_set():
        for _, turn in recorded:
            history.remove(turn)
        for turn in user_turns:
            turn.response = Response.PENDING
        logger.info("started before the gate decided; it looks, so starting over with what it sees")
        return True

    if cancel.is_set():
        voice.stop()
        status = Response.CANCELLED
        if partial.strip():  # cut off while the model was still writing this step
            turn = PalTurn(in_reply_to=[t.id for t in user_turns],
                           utterance=Spoken(actor=Actor.PAL, role=Role.DIRECT, text=partial.strip()))
            _insert_reply(history, turn, user_turns)
            recorded.append((step, turn))

    for step_, turn in recorded:
        words = len(turn.utterance.text.split()) if turn.utterance else 0
        heard = voice.heard((reply_id, step_))
        if cancel.is_set() and turn.utterance and (heard < words or step_ == step):
            turn.cancel(at_word=heard)
            logger.info(f"cut off after {heard} of {words} words: {turn.spoken_text!r}")
        else:
            turn.delivery = Delivery.DONE

    for turn in user_turns:
        turn.response = status
        record("turn", turn)
    _log_timing(start, user_turns, first_text, first_sentence,
                [voice.started_at((reply_id, s)) for s in range(step + 1)])
    for _, turn in recorded:
        record("turn", turn)
    return False


def _insert_reply(history, turn: PalTurn, user_turns: list[UserTurn]):
    """Place a reply turn right after the last turn of this exchange, so a user turn that
    interrupted it (appended meanwhile by the router) comes after it, as it happened."""
    ids = {t.id for t in user_turns}
    belongs = lambda t: t.id in ids or (isinstance(t, PalTurn) and set(t.in_reply_to) & ids)
    last = max((i for i, t in enumerate(history) if belongs(t)), default=len(history) - 1)
    history.insert(last + 1, turn)


def _log_timing(start, user_turns, first_text, first_sentence, audio_starts):
    """One line per reply: where the time between hearing and speaking went. The preview also
    gets the stages before it (marked by listen and the router), for its latency bar."""
    since = lambda t: f"{t - start:.2f}s" if t else "-"
    first_audio = min((t for t in audio_starts if t), default=None)
    waited = start - user_turns[-1].utterance.at.timestamp()
    logger.info(f"timing: waited {waited:.2f}s, first text {since(first_text)}, first sentence {since(first_sentence)}, "
                f"first audio {since(first_audio)}, done {since(time.time())}")
    marks = trace.take(user_turns[-1].utterance.id)
    marks.update(start=start, first_text=first_text, first_sentence=first_sentence, first_audio=first_audio)
    preview.add_latency(user_turns[-1].utterance.text, {k: v for k, v in marks.items() if v})


# Words that open a new clause after a comma; a comma before anything else is not a good place to pause.
CLAUSE_STARTERS = {
    "and", "but", "or", "so", "yet", "because", "since", "while", "although", "though", "if", "unless",
    "when", "where", "which", "who", "that", "then", "as", "after", "before", "until",
    "i", "you", "he", "she", "it", "we", "they", "there", "this", "these", "those",
    "the", "a", "an", "my", "your", "its", "our", "their",
}


class SentenceBuffer:
    """Collects streamed text and hands back whole sentences.

    The first piece of a reply goes out at the first clause break instead, so
    the voice can start sooner: synthesis time grows with length, and waiting
    for the end of a long first sentence delays everything.
    """

    MIN_FIRST_CLAUSE_WORDS = 3

    def __init__(self):
        self.text = ""
        self.first = True

    def feed(self, text: str) -> list[str]:
        self.text += text
        if self.first:
            clause = self._first_clause()
            if clause:
                return [clause]
        # Cut after the last sentence end that is followed by a space (or is the end so far).
        cut = max((m.end() for m in re.finditer(r"[.!?\n]+(\s|$)", self.text)), default=0)
        done, self.text = self.text[:cut], self.text[cut:]
        if done.strip():
            self.first = False
        return _sentences(done)

    def _first_clause(self) -> str | None:
        # A comma only ends a clause when a new one starts after it ("..., but", "..., I"),
        # not between adjectives ("dark, tousled hair") or before a name ("hear you, Job").
        # Wait until the next word has fully arrived before deciding.
        for m in re.finditer(r"[,;:]\s+([A-Za-z']+)(?=[\s,.;:!?])", self.text):
            clause = self.text[:m.start() + 1]
            if re.search(r"[.!?\n]", clause):
                return None  # a whole sentence came first; let the normal path take it
            if len(clause.split()) >= self.MIN_FIRST_CLAUSE_WORDS and m.group(1).lower() in CLAUSE_STARTERS:
                self.text = self.text[m.start() + 1:]
                self.first = False
                return clause.strip()
        return None

    def flush(self) -> list[str]:
        done, self.text = self.text, ""
        return _sentences(done)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


# --- Sinks -------------------------------------------------------------------

def default_sinks(voice: Voice) -> list[Sink]:
    return [tool_sound(voice), log_tools()]


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
            logger.info(f"tool: {event.action.tool}({event.action.args})")
        elif isinstance(event, ToolFinished):
            elapsed_ms = (time.time() - started.pop(event.action.id)) * 1000
            logger.info(f"tool: {event.action.tool} -> {event.action.result[:100]} ({elapsed_ms:.0f}ms)")
        elif isinstance(event, StepDone) and event.refused:
            logger.info("declined the request")

    return sink
