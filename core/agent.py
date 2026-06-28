"""
Agentic decision layer.

The agent follows this loop:
1. Retrieve top-k chunks from the local knowledge base.
2. Ask Gemini to evaluate whether those chunks are sufficient to answer the query.
3. If sufficient   -> answer using local context only.   Source = "Local Knowledge Base".
4. If insufficient -> run a DuckDuckGo web search, fetch full text for top results.
5. Re-evaluate the combined (local + web) context.
6. Generate the final answer with source attribution:
     - "Local Knowledge Base"  (local was sufficient; web never consulted)
     - "Internet Sources"      (local was empty/insufficient AND web filled the gap)
     - "Both"                  (answer materially drew on local AND web context)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from core.gemini_client import GeminiClient, evaluate_context
from core.vector_store import VectorStore
from core.web_search import search_and_fetch

logger = logging.getLogger(__name__)

# Thresholds - tuned for assignment-grade behavior; tweak for production.
LOCAL_TOP_K = 5
WEB_MAX_RESULTS = 5
SIMILARITY_FLOOR = 0.30   # below this, a chunk is treated as noise.
SUFFICIENCY_CONFIDENCE_FLOOR = 0.55


@dataclass
class AgentStep:
    """A single reasoning step the agent took - used for the UI trace panel."""
    step: str
    detail: str


@dataclass
class AgentResult:
    """Final result returned to the UI."""
    answer: str
    source: str                      # "Local Knowledge Base" | "Internet Sources" | "Both"
    local_chunks: List[dict] = field(default_factory=list)
    web_chunks: List[dict] = field(default_factory=list)
    evaluation: dict = field(default_factory=dict)
    trace: List[AgentStep] = field(default_factory=list)


ANSWER_SYSTEM_INSTRUCTION = (
    "You are a careful, factual assistant for a multimodal retrieval-augmented "
    "generation system. Answer the user's query using ONLY the provided CONTEXT. "
    "If the context does not contain the answer, say you don't know - do not "
    "fabricate. Cite sources inline using [source: filename] or [source: web URL] "
    "markers. Be concise but complete."
)


class RAGAgent:
    """Orchestrates retrieval, evaluation, and (optional) web search."""

    def __init__(self, gemini: GeminiClient, vector_store: VectorStore):
        self.gemini = gemini
        self.vs = vector_store

    def run(self, query: str) -> AgentResult:
        trace: List[AgentStep] = []
        result = AgentResult(answer="", source="Local Knowledge Base")

        # ---- Step 1: Local retrieval ----
        trace.append(AgentStep("Local retrieval", f"Querying vector store top-{LOCAL_TOP_K}"))
        local_chunks = self.vs.query(query, self.gemini, top_k=LOCAL_TOP_K)
        # Filter out noise.
        local_chunks = [c for c in local_chunks if c["similarity"] >= SIMILARITY_FLOOR]
        result.local_chunks = local_chunks
        trace.append(
            AgentStep(
                "Local retrieval result",
                f"{len(local_chunks)} chunks above similarity floor {SIMILARITY_FLOOR}",
            )
        )

        # ---- Step 2: Evaluate local sufficiency ----
        if local_chunks:
            local_context = [f"[LOCAL] (source: {c['source']}, sim={c['similarity']:.2f})\n{c['text']}" for c in local_chunks]
            eval_local = evaluate_context(self.gemini, query, local_context)
            result.evaluation = eval_local
            trace.append(
                AgentStep(
                    "Local evaluation",
                    f"sufficient={eval_local.get('sufficient')}, "
                    f"confidence={eval_local.get('confidence')}, "
                    f"reason={eval_local.get('reason')}",
                )
            )

            sufficient = bool(eval_local.get("sufficient"))
            confidence = float(eval_local.get("confidence", 0.0))

            if sufficient and confidence >= SUFFICIENCY_CONFIDENCE_FLOOR:
                # ---- Path A: Local only ----
                trace.append(AgentStep("Decision", "Local context sufficient; answering from local KB."))
                answer = self.gemini.generate(
                    prompt=query,
                    context="\n\n".join(local_context),
                    system_instruction=ANSWER_SYSTEM_INSTRUCTION,
                )
                result.answer = answer
                result.source = "Local Knowledge Base"
                result.trace = trace
                return result
        else:
            trace.append(AgentStep("Local evaluation", "No local chunks retrieved."))
            result.evaluation = {"sufficient": False, "confidence": 0.0, "reason": "no local results"}

        # ---- Step 3: Web search fallback ----
        trace.append(AgentStep("Web search", f"Querying DuckDuckGo top-{WEB_MAX_RESULTS}"))
        web_results = search_and_fetch(query, max_results=WEB_MAX_RESULTS)
        result.web_chunks = web_results
        trace.append(AgentStep("Web search result", f"{len(web_results)} results returned"))

        # Build web context chunks.
        web_context: List[str] = []
        for r in web_results:
            text = r.get("full_text") or r.get("snippet", "")
            if text:
                web_context.append(
                    f"[WEB] (source: {r.get('url', '')}, title: {r.get('title', '')})\n{text}"
                )

        # ---- Step 4: Re-evaluate combined context ----
        combined_context = []
        if local_chunks:
            combined_context.extend(
                f"[LOCAL] (source: {c['source']}, sim={c['similarity']:.2f})\n{c['text']}"
                for c in local_chunks
            )
        combined_context.extend(web_context)

        if combined_context:
            eval_combined = evaluate_context(self.gemini, query, combined_context)
            trace.append(
                AgentStep(
                    "Combined evaluation",
                    f"sufficient={eval_combined.get('sufficient')}, "
                    f"confidence={eval_combined.get('confidence')}",
                )
            )
        else:
            eval_combined = {"sufficient": False, "confidence": 0.0, "reason": "no context"}
            trace.append(AgentStep("Combined evaluation", "No usable context from local or web."))

        # ---- Step 5: Generate final answer ----
        if not combined_context:
            result.answer = (
                "I couldn't find any relevant information in the local knowledge base "
                "or via web search to answer your query. Please try uploading relevant "
                "documents, or rephrase your question."
            )
            result.source = "Local Knowledge Base"
            result.trace = trace
            return result

        answer = self.gemini.generate(
            prompt=query,
            context="\n\n".join(combined_context),
            system_instruction=ANSWER_SYSTEM_INSTRUCTION,
        )
        result.answer = answer

        # ---- Step 6: Determine final source label ----
        local_used = bool(local_chunks)
        web_used = bool(web_context)

        if local_used and web_used:
            # Decide "Both" only if local context was actually used in the answer.
            # We approximate by checking if any local source filename appears in the answer.
            local_sources = {c["source"] for c in local_chunks}
            local_cited = any(src in answer for src in local_sources)
            web_cited = any(r.get("url", "") and r["url"] in answer for r in web_results)
            if local_cited and web_cited:
                result.source = "Both"
            elif web_cited and not local_cited:
                result.source = "Internet Sources"
            elif local_cited and not web_cited:
                result.source = "Local Knowledge Base"
            else:
                # No explicit citation - infer from which had more substance.
                local_chars = sum(len(c["text"]) for c in local_chunks)
                web_chars = sum(len(r.get("full_text") or r.get("snippet", "")) for r in web_results)
                if local_chars > web_chars * 1.5:
                    result.source = "Local Knowledge Base"
                elif web_chars > local_chars * 1.5:
                    result.source = "Internet Sources"
                else:
                    result.source = "Both"
        elif web_used:
            result.source = "Internet Sources"
        else:
            result.source = "Local Knowledge Base"

        trace.append(AgentStep("Final decision", f"Source attribution: {result.source}"))
        result.trace = trace
        return result
