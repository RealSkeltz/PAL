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

