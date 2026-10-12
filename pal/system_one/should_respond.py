from dataclasses import dataclass

from pal.clients import get_client

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
                "The user is speaking to PAL: they call PAL by name to ask or tell it something, "
                "continue a conversation PAL is part of, or give an assistant-style request while "
                "nobody else is around."
            ),
            'can_help': (
                "The speech is not meant for PAL, but the user is stuck on something PAL can clearly "
                "resolve from its memories or the scene, such as looking for an object PAL has seen."
            ),
            'not_for_pal': (
                "The speech is meant for another person, is someone other than the user talking, is "
                "small talk, merely mentions PAL, or is the user thinking aloud about something PAL "
                "cannot help with."
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

# Choices that make PAL speak, with the minimum probability each needs.
MIN_PROB = {'addressed': 0.5, 'warn': 0.5, 'can_help': 0.6}


@dataclass
class Decision:
    speak: bool
    reason: str
    probability: float
    probabilities: dict[str, float]


def should_speak(state, min_prob=MIN_PROB) -> Decision:
    answer = client.decide(state, QUESTIONS[state['event']['type']]).answers['should_speak']
    p = answer.probabilities[answer.choice]
    speak = answer.choice in min_prob and p >= min_prob[answer.choice]
    return Decision(speak, answer.choice, p, answer.probabilities)