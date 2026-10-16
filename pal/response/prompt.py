"""Assemble an agent's system prompt from its context directory.

Each agent owns a `context/` folder of markdown fragments; this stitches the
ones that exist into a single prompt.
"""
from pathlib import Path


def _read(context_dir: Path, name: str) -> str:
    path = context_dir / name
    if not path.exists():
        return ""
    return path.read_text().strip()


def build_system_prompt(context_dir: Path) -> str:
    base = _read(context_dir, "base_prompt.md")
    user_prefs = _read(context_dir, "user_preferences.md")
    domain = _read(context_dir, "domain_knowledge.md")

    sections = [base]
    if user_prefs:
        sections.append("## About the user\n\n" + user_prefs)
    if domain:
        sections.append("## Task context\n\n" + domain)

    return "\n\n".join(sections)
