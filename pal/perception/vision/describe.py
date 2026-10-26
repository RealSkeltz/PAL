"""Put what a look showed into words, so memory knows what was seen.

The images only go to the model with the question they were taken for (and a
follow-up or two). Everything that lasts longer, the memory updaters, the
timeline and recall, works from text. Without a description a look leaves
nothing there but the fact that one happened.
"""
import base64
from pathlib import Path

from pal.llm.base import LLMClient, Turn
from pal.schemas.event import Observation

SYSTEM = (
    "These are moments of what the user sees, from their point of view, oldest first. Write a short note "
    "of what is there: one to three plain sentences, no lists. Be specific and factual: the things that "
    "stand out and where they are, what the user holds or does with their hands, text you can read (quote "
    "it exactly), and the place. Describe other people by appearance and what they do, never by who they "
    "might be. Mention a change between the moments if there is one. Don't guess at what you can't see. "
    "Describe the scene itself, never the viewing of it: no 'I', no camera, images or frames. "
    "You also get what the user had just asked. Make sure the note covers what that question was about, "
    "but describe what is there, not what the question assumes, and don't answer it.\n"
    "Example: \"A white mug with 'tea and mama' and a red heart, held in the user's right hand above a "
    "wooden kitchen table. A laptop is open on the left.\""
)


def describe(client: LLMClient, frames: list[Observation], question: str) -> str:
    """A short description of the frames of one look. `question` is what the user asked, so the
    part that mattered to them isn't left out."""
    images = [base64.b64encode(Path(o.image_path).read_bytes()).decode() for o in frames if o.image_path]
    turn = Turn(role="user", text=f"The user had just asked: {question!r}", images=images)
    return client.complete(SYSTEM, [turn]).text.strip()
