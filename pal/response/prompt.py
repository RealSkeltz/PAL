"""Assemble PAL's system prompt from the markdown fragments in system_prompt/.

The fragments that exist are stitched into a single prompt.
"""
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "system_prompt"


def _read(context_dir: Path, name: str) -> str:
    path = context_dir / name
    if not path.exists():
        return ""
    return path.read_text().strip()


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
