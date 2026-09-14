"""Plain message types passed between the agent loop and the LLM."""

from typing import Any, Dict, List, Optional


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(content={self.content!r})"


class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__("system", content)


class UserMessage(Message):
    def __init__(self, content: str):
        super().__init__("user", content)


class AssistantMessage(Message):
    def __init__(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None):
        super().__init__("assistant", content)
        self.tool_calls = tool_calls or []

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data


class ToolMessage(Message):
    def __init__(self, content: str, tool_call_id: str, name: Optional[str] = None):
        super().__init__("tool", content)
        self.tool_call_id = tool_call_id
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data
