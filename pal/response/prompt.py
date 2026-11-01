"""Assemble PAL's system prompt from the markdown fragments in system_prompt/, and the
note that ends each agent turn.

The fragments that exist are stitched into a single prompt. What PAL knows and
what it hasn't seen yet change all the time, so they go at the end of the
conversation instead: in the system prompt, every change would make the
cached conversation after it useless.
"""
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "system_prompt"


def _read(context_dir: Path, name: str) -> str:
    path = context_dir / name
    if not path.exists():
        return ""
    return path.read_text().strip()


def about_user(context_dir: Path = PROMPT_DIR) -> str:
    """Who the user is, for anything that needs to recognise them (e.g. by name)."""
    return _read(context_dir, "user_preferences.md")


def build_system_prompt(context_dir: Path = PROMPT_DIR) -> str:
    base = _read(context_dir, "base_prompt.md")
    user_prefs = _read(context_dir, "user_preferences.md")
    domain = _read(context_dir, "domain_knowledge.md")

    sections = [base]
    if user_prefs:
        sections.append("## About the user\n\n" + user_prefs)
    if domain:
        sections.append("## Task context\n\n" + domain)

    return "\n\n".join(sections)


def _in_view_text(in_view: str) -> str:
    return f"In view right now (object recognition):\n{in_view}\n\n" if in_view else ""


def respond_note(view: str, unseen: str, in_view: str = "") -> str:
    """Ends a reply turn: what happened around the user that wasn't said to PAL, what's in view,
    and what PAL knows."""
    since = f"Not said to you, since your last turn:\n{unseen}\n\n" if unseen else ""
    return (f"[For context; answer the user's message above.\n\n{since}{_in_view_text(in_view)}"
            f"What you know (keys in brackets):\n{view}\n\n"
            "If this exchange changes something lasting or the situation, update your memory after answering.]")


def reflect_note(view: str, unseen: str, in_view: str = "") -> str:
    """Ends a reflect turn: what's new, what's in view, what PAL knows, and what to do with it."""
    return ("[A quiet moment. Nobody is waiting for an answer, and nothing you write is said aloud.\n\n"
            f"Since you last looked:\n{unseen}\n\n{_in_view_text(in_view)}What you know (keys in brackets):\n{view}\n\n"
            "Bring your memory up to date with your tools, each thing in its own place:\n"
            "- situation: only where the user is, what they're doing, and who is physically there, in a few words\n"
            "- remember: lasting facts, such as who someone is to the user ('Mark: the user's brother') or a preference\n"
            "- add_goal / update_goal: what the user is trying to get done, with deadlines ('frost the cake before six')\n"
            "- log_episode: one sentence for anything worth remembering later\n"
            "- update_object: a name or description for someone or something in view, when the notes make clear who or what it is\n"
            "- ask_user: only if something you noticed is worth checking with the user; it's said aloud\n"
            "Change only what the new notes support and leave the rest; revise an item by its key rather than "
            "adding it again; when nothing needs changing, call nothing. Describe the user's world, not their use "
            "of you. A single hint is a guess with low confidence. Reply with only '-'.]")
