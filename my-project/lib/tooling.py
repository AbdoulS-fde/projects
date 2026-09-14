"""`@tool` decorator that exposes a plain function to the LLM as a callable tool."""

import inspect
from typing import Any, Callable, Dict, Optional, get_type_hints

_JSON_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool:
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").strip()
        self.parameters = self._build_schema(func)

    @staticmethod
    def _build_schema(func: Callable) -> Dict[str, Any]:
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        properties: Dict[str, Any] = {}
        required = []
        for param_name, param in signature.parameters.items():
            param_type = hints.get(param_name, str)
            properties[param_name] = {"type": _JSON_TYPE_MAP.get(param_type, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        return {"type": "object", "properties": properties, "required": required}

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def run(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None):
    def wrap(f: Callable) -> Tool:
        return Tool(f, name=name, description=description)

    if func is not None:
        return wrap(func)
    return wrap
