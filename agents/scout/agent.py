import os
from pathlib import Path

from pal.agent import Pal
from pal.context import build_system_prompt
from agents.scout.tools import ScoutTools

CONTEXT_DIR = Path(__file__).parent / "context"

# Override per run: PAL_MODEL=anthropic:claude-opus-5 python main.py
DEFAULT_MODEL = "ollama:qwen3.5:9b"

class Scout(Pal):
    tool_handler_cls = ScoutTools

    def __init__(self):
        super().__init__(
            system_prompt=build_system_prompt(CONTEXT_DIR),
            model=os.environ.get("PAL_MODEL", DEFAULT_MODEL),
        )
