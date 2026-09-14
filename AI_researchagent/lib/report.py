"""Phase 4: turn accumulated evidence (RAG hits + web hits + evaluation notes) into a
structured, cited report instead of a raw string."""

import json
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from lib.llm import LLM
from lib.messages import SystemMessage, UserMessage

REPORT_SYSTEM_PROMPT = (
    "You are UdaPlay, a video game research assistant. Using ONLY the evidence provided, produce a "
    "structured report answering the user's question. Tag every fact's origin as 'internal' if it "
    "came from the local knowledge base results, or 'web' if it came from the web search results. "
    "If the evidence is insufficient to answer confidently, say so plainly in the answer and lower "
    "the confidence score accordingly.\n\n"
    "Before answering, check two things:\n"
    "1. Does the question clearly name or identify a specific video game? If it's vague or "
    "underspecified (e.g. 'tell me about the game' with no title given), do NOT guess which game is "
    "meant. Instead answer that you could not identify a specific game from the question and ask for "
    "clarification, and set confidence to 0.2 or lower.\n"
    "2. Does the evidence actually describe a real video game that matches the question? If the "
    "evidence is off-topic or describes something that is not a video game (e.g. a generic activity, "
    "meme, or unrelated topic that merely shares a word with the question), do NOT present it as the "
    "answer. Instead say the evidence found does not answer the question, and set confidence to 0.2 "
    "or lower.\n"
    "Never fabricate an answer beyond what the evidence supports just because some evidence was "
    "retrieved.\n\n"
    "Respond with ONLY a JSON object matching this schema (no markdown fences, no extra text):\n"
    "{\n"
    '  "answer": str,                 // direct, concise answer to the question\n'
    '  "supporting_details": [str],   // short bullet-point facts backing the answer\n'
    '  "sources": [\n'
    "    {\n"
    '      "origin": "internal" | "web",\n'
    '      "title": str,\n'
    '      "url": str | null          // omit/null for internal knowledge base entries\n'
    "    }\n"
    "  ],\n"
    '  "confidence": float             // overall confidence in the answer, between 0.0 and 1.0\n'
    "}"
)


class Source(BaseModel):
    origin: Literal["internal", "web"]
    title: str
    url: Optional[str] = None


class ResearchReport(BaseModel):
    """Structured output of the REPORT step: a direct answer with supporting details and cited sources."""

    question: str
    answer: str
    supporting_details: List[str] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    confidence: float = 0.0

    def to_text(self) -> str:
        """Render the report as human-readable text for notebook/CLI display."""
        lines = [f"Q: {self.question}", "", f"Answer: {self.answer}"]

        if self.supporting_details:
            lines.append("\nSupporting details:")
            lines.extend(f"  - {detail}" for detail in self.supporting_details)

        if self.sources:
            lines.append("\nSources:")
            for source in self.sources:
                label = "internal knowledge" if source.origin == "internal" else "web search"
                suffix = f" ({source.url})" if source.url else ""
                lines.append(f"  - [{label}] {source.title}{suffix}")

        lines.append(f"\nConfidence: {self.confidence:.2f}")
        return "\n".join(lines)


def generate_report(
    llm: LLM,
    question: str,
    retrieval_json: Optional[str] = None,
    web_json: Optional[str] = None,
) -> ResearchReport:
    """Ask the LLM to synthesize the accumulated evidence into a structured, cited ResearchReport."""
    evidence_parts = []
    if retrieval_json:
        evidence_parts.append(f"LOCAL KNOWLEDGE BASE RESULTS (origin=internal):\n{retrieval_json}")
    if web_json:
        evidence_parts.append(f"WEB SEARCH RESULTS (origin=web):\n{web_json}")
    evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "No evidence was found."

    messages = [
        SystemMessage(REPORT_SYSTEM_PROMPT),
        UserMessage(f"Question: {question}\n\n{evidence_text}"),
    ]
    response = llm.invoke(messages)

    try:
        payload = json.loads(response.content)
        return ResearchReport(question=question, **payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Model didn't return valid JSON; fall back to a plain, unstructured answer rather than crashing.
        return ResearchReport(question=question, answer=response.content, confidence=0.0)
