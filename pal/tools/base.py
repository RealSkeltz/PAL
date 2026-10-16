"""Tool plumbing: declare a tool by decorating the method that implements it.

Specs are provider-neutral (`pal.llm.ToolSpec`) — each LLM backend translates
them into its own schema dialect.
"""
from pal.llm.base import ToolSpec


def tool(description: str, params: dict | None = None, required: list[str] | None = None):
    """Mark a handler method as a callable tool and attach its spec."""
    def decorator(fn):
        fn.tool_spec = ToolSpec(
            name=fn.__name__,
            description=description,
            properties=params or {},
            required=required or [],
        )
        return fn
    return decorator


class ToolHandler:
    """Collects every @tool method on the class and exposes them to the model."""

    def __init__(self, pal):
        self.pal = pal  # reference back so tools can read last_image, conversation, etc.
        self._tools = {
            name: getattr(self, name)
            for name in dir(type(self))
            if hasattr(getattr(type(self), name, None), "tool_spec")
        }

    @property
    def specs(self) -> list[ToolSpec]:
        return [fn.tool_spec for fn in self._tools.values()]

    def dispatch(self, name: str, arguments: dict) -> str:
        """Execute a tool call and return a string result for the model."""
        handler = self._tools.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            return handler(**arguments)
        except Exception as e:
            return f"Error: {e}"
