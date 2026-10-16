from dataclasses import dataclass

from pal.llm import get_client

STATE_DESCRIPTION = (
    "The state describes a moment for a user wearing PAL, a voice assistant with a camera and "
    "microphone. 'watching_for' lists things the user asked PAL to watch for, 'memories' are "
    "things PAL observed earlier that may be relevant, 'recent' is the conversation so far, "
    "and 'event' is what just happened. "
)

SPEECH_QUESTIONS = {
    'should_speak': {
        'type': 'choice',
        'instructions': STATE_DESCRIPTION + (
            "The event is speech that was just heard; 'speaker' is who said it and 'scene' is what "
            "the camera currently sees. Should PAL speak in response to this speech?"
        ),
        'criteria': {
            'addressed': (
                "The user is speaking to PAL: they call PAL by name, continue a conversation PAL is "
                "part of, or ask a question or give a request a voice assistant would handle, such as "
                "asking the time, what they are looking at, where something is, or to set a timer or "
                "change the volume."
            ),
            'can_help': (
                "The speech is not directed at PAL, for example the user thinking aloud or talking to "
                "someone else, but the user is stuck on something PAL can clearly resolve from its "
                "memories or the scene, such as looking for an object PAL has seen."
            ),
            'not_for_pal': (
                "The speech is clearly meant for another person, is small talk or an exclamation, "
                "merely mentions PAL, or is the user thinking aloud about something PAL cannot help with."
            ),
        },
    },
}

SCENE_QUESTIONS = {
    'should_speak': {
        'type': 'choice',
        'instructions': STATE_DESCRIPTION + (
            "The event is a new description of what the camera sees. Should PAL say something to "
            "the user about it, without being asked?"
        ),
        'criteria': {
            'warn': (
                "The scene shows a real danger to the user or others, or something listed in "
                "'watching_for' is happening."
            ),
            'can_help': (
                "There is no danger, but the user is visibly stuck on something PAL can clearly "
                "resolve from its memories, such as a door they cannot open while PAL knows where "
                "their keys are."
            ),
            'nothing': (
                "Nothing in the scene needs the user's attention; speaking up would only interrupt."
            ),
        },
    },
}

QUESTIONS = {'speech': SPEECH_QUESTIONS, 'scene': SCENE_QUESTIONS}

client = get_client('typesafe:tev1:4b')

@dataclass
class Decision:
    speak: bool
    reason: str
    probability: float
    probabilities: dict[str, float]


def get_should_answer_noul(state) -> Decision:
    answer = client.decide(state, QUESTIONS[state['event']['type']]).answers['should_speak']
    return answer


def make_state(event, scene=None, watching_for=(), memories=(), recent=()):
    """Stable context first, the new event last."""
    state = {'watching_for': list(watching_for), 'memories': list(memories), 'recent': list(recent)}
    if scene is not None:
        state['scene'] = scene
    state['event'] = event
    return state


def heard(text, scene, speaker='user', **context):
    return make_state({'type': 'speech', 'speaker': speaker, 'text': text}, scene=scene, **context)


def seen(description, **context):
    return make_state({'type': 'scene', 'description': description}, **context)

