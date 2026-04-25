from pathlib import Path

CONTEXT_DIR = Path(__file__).parent / "scout"


def _read(name: str) -> str:
    path = CONTEXT_DIR / name
    if not path.exists():
        return ""
    return path.read_text().strip()


def build_system_prompt() -> str:
    base = _read("base_prompt.md")
    user_prefs = _read("user_preferences.md")
    domain = _read("domain_knowledge.md")

    sections = [base]
    if user_prefs:
        sections.append("## About the user\n\n" + user_prefs)
    if domain:
        sections.append("## Task context\n\n" + domain)

    return "\n\n".join(sections)