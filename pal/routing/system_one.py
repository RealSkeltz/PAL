from dataclasses import dataclass

from pal.llm import get_client

# Every token here is read on every decision (about 1.4 ms each on the dev Mac), so keep these short.
STATE_DESCRIPTION = (
    "The state is a moment for someone wearing Buddy, a voice assistant with a camera. "
    "'situation' is what Buddy knows about the surroundings, 'recent' is the conversation so far "
    "and 'event' is what just happened. "
)

SPEECH_QUESTIONS = {
    'should_speak': {
        'type': 'choice',
        'instructions': STATE_DESCRIPTION + "The event is speech the user just said. Should Buddy respond?",
        'criteria': {
            'addressed': "Said to Buddy: a question or request an assistant handles, or a follow-up to Buddy. "
                         "Short questions like 'and now?' or 'what about this one?' right after Buddy answered continue that conversation.",
            'can_help': "Not said to Buddy, but Buddy can clearly solve what the user is stuck on.",
            'not_for_pal': "Said to someone else, small talk, an exclamation, or thinking aloud.",
        },
    },
}

SCENE_QUESTIONS = {
    'should_speak': {
        'type': 'choice',
        'instructions': STATE_DESCRIPTION + "The event is what the camera now sees. Should Buddy speak up unasked?",
        'criteria': {
            'warn': "A real danger, or something in 'watching_for' is happening.",
            'can_help': "The user is visibly stuck on something Buddy can clearly solve.",
            'nothing': "Nothing needs the user's attention.",
        },
    },
}

VISION_QUESTIONS = {
    'needs_vision': {
        'type': 'choice',
        # Biased towards looking: a needless look costs little, a missed one gives a wrong answer.
        'instructions': STATE_DESCRIPTION + "The event is a request to Buddy. Would seeing what is in front of the user help answer it?",
        'criteria': {
            'look': "It touches the physical world around the user: what they are doing, holding or looking at, people, "
                    "animals or things nearby, where something is, or 'this', 'that', 'here', 'check it'. Also reading, "
                    "identifying, checking or describing something. When unsure, look.",
            'no_look': "Only about knowledge, the conversation itself, Buddy, or a tool such as notes or timers.",
        },
    },
}

client = get_client('typesafe:tev1:4b')

@dataclass
class Decision:
    yes: bool
    reason: str
    probability: float
    probabilities: dict[str, float]


def ask(state, questions):
    """Answer several questions about one state in a single model call."""
    return client.decide(state, questions).answers


def make_state(event, scene=None, watching_for=(), memories=(), recent=(), situation=None):
    """Stable context first, the new event last. Empty fields are left out; they only cost tokens."""
    state = {'situation': situation, 'watching_for': list(watching_for), 'memories': list(memories),
             'recent': list(recent), 'scene': scene}
    state = {k: v for k, v in state.items() if v}
    state['event'] = event
    return state


def heard(text, scene, speaker='user', **context):
    return make_state({'type': 'speech', 'speaker': speaker, 'text': text}, scene=scene, **context)


def seen(description, **context):
    return make_state({'type': 'scene', 'description': description}, **context)

