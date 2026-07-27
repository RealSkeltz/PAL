from pathlib import Path

from pal.agent import Pal
from pal.context import build_system_prompt
from agents.scout.tools import ScoutTools
from pal.config import MODELS_DIR

CONTEXT_DIR = Path(__file__).parent / "context"

class Scout(Pal):
    tool_handler_cls = ScoutTools

    def __init__(self):
        super().__init__(
            system_prompt=build_system_prompt(CONTEXT_DIR),
            # "provider:model" — swap to "anthropic:claude-opus-5" to run on Claude.
            model="ollama:qwen3.5:9b",
        )
