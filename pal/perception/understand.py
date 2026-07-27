"""One-shot image captioning, on whichever backend the caller passes in."""
import base64

import cv2

from pal.llm import Turn


def understand_image(image, context: str, client) -> str:
    """Describe `image` in a sentence. `client` may be any LLMClient."""
    turn = Turn(
        role="user",
        text=f"Describe the image in one concise sentence. There is a {context} in the image.",
        images=[numpy_to_base64(image)],
    )
    return client.complete(system=None, conversation=[turn]).text


def numpy_to_base64(img) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")
