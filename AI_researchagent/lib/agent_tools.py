"""Agent-facing tools for Phase 2: game retrieval, retrieval evaluation, and web search fallback."""

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tavily import TavilyClient

from lib.tooling import tool
from lib.vector_store import VectorStoreManager

load_dotenv()

_store: Optional[VectorStoreManager] = None
_tavily_client: Optional[TavilyClient] = None


def configure_vector_store(store: VectorStoreManager) -> None:
    """Inject an already-constructed VectorStoreManager (e.g. the one built in the notebook) for tools to reuse."""
    global _store
    _store = store


def _get_store() -> VectorStoreManager:
    global _store
    if _store is None:
        _store = VectorStoreManager(base_url=os.getenv("OPENAI_BASE_URL"))
    return _store


def _get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


@tool
def retrieve_game(question: str, k: int = 5) -> str:
    """Semantically search the games vector store for the top-k games most relevant to a natural-language question.

    Returns a JSON string: {"question": str, "results": [{"id", "title", "document", "metadata", "distance"}, ...]}.
    Lower distance means a closer match.
    """
    raw = _get_store().query(question, k=k)
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    results = [
        {
            "id": id_,
            "title": (meta or {}).get("title"),
            "document": doc,
            "metadata": meta,
            "distance": distance,
        }
        for id_, doc, meta, distance in zip(ids, documents, metadatas, distances)
    ]
    return json.dumps({"question": question, "results": results})


@tool
def evaluate_retrieval(question: str, retrieved_results: str, distance_threshold: float = 0.35) -> str:
    """Judge whether the results from `retrieve_game` are sufficient to answer the question.

    Applies a threshold on the best (lowest) distance score rather than a second LLM call, so the
    verdict is fast, deterministic, and cheap. `retrieved_results` must be the JSON string returned
    by `retrieve_game`. Returns a JSON string: {"sufficient": bool, "confidence": float, "reasoning": str}.
    """
    try:
        results: List[Dict[str, Any]] = json.loads(retrieved_results).get("results", [])
    except (json.JSONDecodeError, AttributeError):
        results = []

    distances = [r["distance"] for r in results if r.get("distance") is not None]
    if not distances:
        verdict = {
            "sufficient": False,
            "confidence": 0.0,
            "reasoning": "No retrieved results to evaluate.",
        }
        return json.dumps(verdict)

    best_distance = min(distances)
    confidence = round(max(0.0, min(1.0, 1 - best_distance)), 4)
    sufficient = best_distance <= distance_threshold
    reasoning = (
        f"Best match distance {best_distance:.4f} is "
        f"{'within' if sufficient else 'above'} the {distance_threshold} threshold for question "
        f"'{question}', so local evidence is {'sufficient' if sufficient else 'insufficient'}."
    )
    verdict = {"sufficient": sufficient, "confidence": confidence, "reasoning": reasoning}
    return json.dumps(verdict)


@tool
def game_web_search(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily as fallback evidence when local retrieval is insufficient.

    Returns a JSON string: {"query": str, "results": [{"title", "url", "snippet"}, ...]}.
    """
    response = _get_tavily_client().search(query=query, max_results=max_results)
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": (item.get("content") or "").strip()[:500],
        }
        for item in response.get("results", [])
    ]
    return json.dumps({"query": query, "results": results})


AGENT_TOOLS = [retrieve_game, evaluate_retrieval, game_web_search]
