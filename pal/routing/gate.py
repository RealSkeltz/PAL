"""Decisions about an utterance: is it for PAL, and does answering it need the camera?

Both are System One questions on the same state, so they go out in one model
call. The cheap shortcuts (called by name, or right after PAL spoke) replace
only the first question; the camera question is always asked.
"""
import re
import time

from pal.routing.system_one import SPEECH_QUESTIONS, VISION_QUESTIONS, Decision, ask, heard
from pal.schemas.turn import ConversationTurn, PalTurn

CONVERSATION_WINDOW_SECONDS = 10  # counted from when Buddy finished speaking; follow-ups often come a few seconds later

# Choices that count as "yes", with the probability each needs.
MIN_PROB = {
    'speech': {'addressed': 0.0, 'can_help': 0.0},
    'vision': {'look': 0.65},  # real 'look' questions score 0.7+; borderline ones attached frames needlessly
    'scene': {'warn': 0.5, 'can_help': 0.6},
}


def addressed_by_name(input_text):
    return re.search(r"\bbuddy\b", input_text, re.IGNORECASE)


def within_conversation_window(last_spoke_at):
    return (time.time() - last_spoke_at) < CONVERSATION_WINDOW_SECONDS


def assess(input_text, history: list[ConversationTurn], ask_speak: bool,
           situation: str | None = None) -> tuple[Decision | None, Decision]:
    """Returns (should PAL speak, or None if not asked; does it need to look)."""
    recent = [
        f"{'assistant' if isinstance(t, PalTurn) else 'user'}: {t.spoken_text if isinstance(t, PalTurn) else t.utterance.text}"
        for t in history[-4:] if t.utterance
    ]
    questions = {**VISION_QUESTIONS, **(SPEECH_QUESTIONS if ask_speak else {})}
    answers = ask(heard(input_text, scene=None, recent=recent, situation=situation), questions)
    speak = _decision(answers['should_speak'], MIN_PROB['speech']) if ask_speak else None
    return speak, _decision(answers['needs_vision'], MIN_PROB['vision'])


def _decision(answer, thresholds) -> Decision:
    p = answer.probabilities[answer.choice]
    yes = answer.choice in thresholds and p >= thresholds[answer.choice]
    return Decision(yes, answer.choice, p, answer.probabilities)
