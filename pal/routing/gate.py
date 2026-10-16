import re
import time

from pal.routing.system_one import Decision, get_should_answer_noul, heard
from pal.schemas.turn import ConversationTurn, PalTurn

CONVERSATION_WINDOW_SECONDS = 5

MIN_PROB = {
    'speech': {'addressed': 0.0, 'can_help': 0.0},
    'scene': {'warn': 0.5, 'can_help': 0.6},
}


def addressed_by_name(input_text):
    return re.search(r"\bpal\b", input_text, re.IGNORECASE)

def within_conversation_window(last_spoke_at):
    return (time.time() - last_spoke_at) < CONVERSATION_WINDOW_SECONDS

def should_speak(input_text, scene, history: list[ConversationTurn]) -> Decision:
    recent = [
        f"{'assistant' if isinstance(t, PalTurn) else 'user'}: {t.spoken_text if isinstance(t, PalTurn) else t.utterance.text}"
        for t in history[-4:] if t.utterance
    ]
    answer = get_should_answer_noul(heard(input_text, scene=scene, recent=recent))
    p = answer.probabilities[answer.choice]
    thresholds = MIN_PROB['speech']
    speak = answer.choice in thresholds and p >= thresholds[answer.choice]
    return Decision(speak, answer.choice, p, answer.probabilities)


