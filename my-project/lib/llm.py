"""Thin wrapper around an OpenAI-compatible chat completions endpoint (e.g. the Vocareum proxy)."""

import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from lib.messages import AssistantMessage, Message
from lib.tooling import Tool

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


class LLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def invoke(self, messages: List[Message], tools: Optional[List[Tool]] = None) -> AssistantMessage:
        kwargs = {}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t.to_dict()} for t in tools]

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[m.to_dict() for m in messages],
            **kwargs,
        )
        choice = response.choices[0].message
        # Keep tool_calls in the exact OpenAI wire shape so replaying them back as history works.
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (choice.tool_calls or [])
        ]
        return AssistantMessage(content=choice.content or "", tool_calls=tool_calls)

