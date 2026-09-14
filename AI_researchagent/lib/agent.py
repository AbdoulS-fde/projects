"""Phase 3: state-machine orchestration wiring the Phase 2 tools into a single research agent."""

import json
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from lib.agent_tools import evaluate_retrieval, game_web_search, retrieve_game
from lib.llm import LLM
from lib.report import ResearchReport, generate_report
from lib.vector_store import VectorStoreManager


class State(Enum):
    RETRIEVE = auto()
    EVALUATE = auto()
    WEB_SEARCH = auto()
    PERSIST = auto()
    REPORT = auto()
    DONE = auto()


class ResearchAgent:
    """Runs Start -> Retrieve -> Evaluate -> (Web Search -> Persist -> Retrieve) -> Report."""

    def __init__(
        self,
        store: VectorStoreManager,
        llm: Optional[LLM] = None,
        k: int = 5,
        distance_threshold: float = 0.35,
        max_iterations: int = 10,
    ):
        self.store = store
        self.llm = llm or LLM(temperature=0.2)
        self.k = k
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations

    def _persist_web_result(self, question: str, web_result: Dict[str, Any]) -> int:
        """Upsert web search hits into the vector store so future queries hit RAG first (long-term memory)."""
        ids, documents, metadatas = [], [], []
        for item in web_result.get("results", []):
            if not item.get("snippet"):
                continue
            doc_id = f"web::{abs(hash((question, item['url'])))}"
            ids.append(doc_id)
            documents.append(f"{item['title']}. {item['snippet']}")
            metadatas.append(
                {
                    "title": item.get("title", ""),
                    "source": "web",
                    "url": item.get("url", ""),
                    "origin_question": question,
                }
            )
        if ids:
            self.store.add_documents(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def answer(self, question: str) -> Dict[str, Any]:
        """Run the full state machine for a single question and return a trace plus a ResearchReport."""
        trace: List[str] = []
        state = State.RETRIEVE
        iterations = 0
        evidence_sources: List[str] = []
        retrieval_json: Optional[str] = None
        web_json: Optional[str] = None
        used_web_search = False

        while state != State.DONE and iterations < self.max_iterations:
            iterations += 1

            if state == State.RETRIEVE:
                trace.append(f"[{iterations}] RETRIEVE: querying vector store for '{question}'")
                retrieval_json = retrieve_game.run(question=question, k=self.k)
                state = State.EVALUATE

            elif state == State.EVALUATE:
                verdict = json.loads(
                    evaluate_retrieval.run(
                        question=question,
                        retrieved_results=retrieval_json,
                        distance_threshold=self.distance_threshold,
                    )
                )
                trace.append(
                    f"[{iterations}] EVALUATE: sufficient={verdict['sufficient']} confidence={verdict['confidence']}"
                )
                if verdict["sufficient"]:
                    evidence_sources.append("local knowledge base")
                    state = State.REPORT
                elif not used_web_search:
                    state = State.WEB_SEARCH
                else:
                    # Already fell back to the web once; report with whatever evidence we have.
                    state = State.REPORT

            elif state == State.WEB_SEARCH:
                trace.append(f"[{iterations}] WEB_SEARCH: falling back to Tavily for '{question}'")
                web_json = game_web_search.run(query=question, max_results=5)
                evidence_sources.append("web search")
                used_web_search = True
                state = State.PERSIST

            elif state == State.PERSIST:
                added = self._persist_web_result(question, json.loads(web_json))
                trace.append(f"[{iterations}] PERSIST: upserted {added} new document(s) into long-term memory")
                state = State.RETRIEVE  # re-check the (now updated) local store before reporting

            elif state == State.REPORT:
                state = State.DONE

        report: ResearchReport = generate_report(self.llm, question, retrieval_json, web_json)
        return {
            "question": question,
            "report": report,
            "sources": sorted(set(evidence_sources)) or ["none found"],
            "trace": trace,
        }
