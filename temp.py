# ---------------------------------------------------------------------
# RagAgent
# ---------------------------------------------------------------------

class RagAgent:
    """
    RAG agent node.

    Input:
      state["rag_request"] as RagRequest or dict with keys:
        - query (str)
        - max_chunks/top_k (int)
        - sources (list[str])
        - context_type (str)

    Output:
      state["rag_status"] = "done" or "failed"
      state["rag_result"] = {
          "used_query": str,
          "chunks": [{"text":..., "source":..., "doc_id":..., "score":...}, ...],
          "notes": str | None,
          "sources_used": list[str] | None
      }
      state["rag_request"] cleared on completion/failure
    """

    def __init__(self, llm: Any, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    def __call__(self, state: GraphState) -> GraphState:
        session_id = state.get("session_id", "unknown")
        logger.info("RagAgent invoked", extra={"session_id": session_id})

        rag_req = state.get("rag_request")
        if not rag_req:
            logger.warning("No rag_request in state", extra={"session_id": session_id})
            state["rag_status"] = "failed"
            state["rag_result"] = None
            state["rag_request"] = None
            state["rag_observations"] = (state.get("rag_observations") or []) + ["Missing rag_request"]
            return state

        # Normalize request
        query, top_k, sources, context_type = self._normalize_request(rag_req)

        if not query.strip():
            state["rag_status"] = "failed"
            state["rag_result"] = None
            state["rag_request"] = None
            state["rag_observations"] = (state.get("rag_observations") or []) + ["Empty rag_request.query"]
            return state

        logger.info(
            "RAG request",
            extra={"session_id": session_id, "query": query[:120], "top_k": top_k, "sources": sources, "context_type": context_type},
        )

        if not _rag_available or _rag_retriever is None:
            msg = "RAG backend not available (FAISS/embeddings not initialized)"
            logger.error(msg, extra={"session_id": session_id})
            state["rag_status"] = "failed"
            state["rag_result"] = None
            state["rag_request"] = None
            state["rag_observations"] = (state.get("rag_observations") or []) + [msg]
            return state

        # Optionally optimize query using LLM (still local via Server A proxy)
        optimized_query = self._optimize_query(query=query, context_type=context_type)

        # Retrieve
        try:
            raw_results = self._search(optimized_query, top_k=top_k)
            chunks = self._format_results(raw_results, sources=sources)

            state["rag_result"] = {
                "used_query": optimized_query,
                "chunks": chunks,
                "notes": f"Retrieved {len(chunks)} chunks for query.",
                "sources_used": sources,
            }
            state["rag_status"] = "done"
            state["rag_request"] = None

            logger.info(
                "RAG completed",
                extra={"session_id": session_id, "n_chunks": len(chunks)},
            )
            return state

        except Exception as e:
            err = f"RAG retrieval failed: {str(e)}"
            logger.error(err, extra={"session_id": session_id})
            logger.debug("RAG retrieval traceback:\n%s", traceback.format_exc())

            state["rag_status"] = "failed"
            state["rag_result"] = None
            state["rag_request"] = None
            state["rag_observations"] = (state.get("rag_observations") or []) + [err]
            return state

    # -----------------------
    # Request normalization
    # -----------------------

    def _normalize_request(
        self, req: Union[RagRequest, Dict[str, Any]]
    ) -> Tuple[str, int, List[str], str]:
        if is_dataclass(req):
            reqd = asdict(req)
        elif isinstance(req, dict):
            reqd = req
        else:
            reqd = {"query": str(req)}

        query = str(reqd.get("query", "") or "")

        # support either max_chunks or top_k
        top_k = reqd.get("max_chunks") or reqd.get("top_k") or DEFAULT_TOP_K
        try:
            top_k = int(top_k)
        except Exception:
            top_k = DEFAULT_TOP_K

        sources = reqd.get("sources") or DEFAULT_SOURCES
        if not isinstance(sources, list):
            sources = [str(sources)]

        context_type = str(reqd.get("context_type") or "general")

        return query, top_k, sources, context_type

    # -----------------------
    # Query optimization (local LLM)
    # -----------------------

    def _optimize_query(self, *, query: str, context_type: str) -> str:
        """
        Uses local LLM to expand/refine query. Falls back safely on errors.
        """
        # Keep optimization short & deterministic
        system_prompt = (
            "You optimize search queries for technical documentation retrieval.\n"
            "Return ONLY the optimized query text, no quotes, no commentary.\n"
            "Keep it under ~200 characters if possible.\n"
        )

        context_guidance = ""
        if context_type == "error_solution":
            context_guidance = "Focus on error strings, tool names, flags, and fixes."
        elif context_type == "how_to":
            context_guidance = "Focus on procedural steps, CLI commands, and required options."
        elif context_type == "conceptual":
            context_guidance = "Focus on definitions, explanations, and key terms."

        messages = [
            SystemMessage(content=f"{system_prompt}\n{context_guidance}".strip()),
            HumanMessage(content=f"Original query: {query}\nOptimized query:"),
        ]

        try:
            resp = self.llm.invoke(messages)
            if isinstance(resp, AIMessage):
                out = (resp.content or "").strip()
            else:
                out = str(resp).strip()

            out = self._post_process_query(out, query)
            return out if out else query

        except Exception as e:
            logger.warning("Query optimization failed; using expanded fallback", extra={"error": str(e)})
            return self._fallback_expand_query(query, context_type=context_type)

    def _post_process_query(self, optimized: str, original: str) -> str:
        # Remove newlines + excessive spaces
        optimized = re.sub(r"\s+", " ", optimized).strip()
        if not optimized:
            return ""

        # Ensure original tokens remain (best-effort)
        orig_terms = set(re.sub(r"\s+", " ", original.lower()).split())
        opt_terms = set(optimized.lower().split())
        missing = [t for t in orig_terms if t not in opt_terms]
        if missing:
            optimized = (optimized + " " + " ".join(missing)).strip()

        # Cap length
        return optimized[:220]

    def _fallback_expand_query(self, query: str, context_type: str) -> str:
        expansions = {
            "error_solution": ["error", "issue", "problem", "solution", "fix", "troubleshooting"],
            "how_to": ["guide", "tutorial", "steps", "procedure", "how to", "instructions"],
            "conceptual": ["overview", "explanation", "understanding", "basics", "fundamentals"],
            "general": ["documentation", "manual", "reference"],
        }
        tech = {
            "qnn": ["Qualcomm Neural Network", "QNN", "HTP"],
            "snpe": ["Snapdragon Neural Processing Engine", "SNPE", "DLC"],
            "tflite": ["TensorFlow Lite", "TFLite"],
            "neuropilot": ["NeuroPilot", "MDLA"],
            "convert": ["converter", "export", "compile"],
            "profile": ["profiling", "benchmark", "latency"],
            "htp": ["Hexagon", "HTP"],
        }

        terms = query.lower().split()
        out = list(dict.fromkeys(terms))  # unique, preserve order
        out.extend(expansions.get(context_type, expansions["general"]))
        for t in terms:
            if t in tech:
                out.extend(tech[t])

        # unique again
        out2: List[str] = []
        seen = set()
        for t in out:
            tl = t.lower()
            if tl not in seen:
                out2.append(t)
                seen.add(tl)

        return " ".join(out2)[:220]

    # -----------------------
    # Retrieval + formatting
    # -----------------------

    def _search(self, query: str, top_k: int) -> Sequence[Any]:
        """
        Calls the FAISS retriever. We keep this generic because different retrievers return different shapes.
        Must return an iterable of results.
        """
        # Common patterns:
        # - retriever.search(query, top_k=top_k)
        # - retriever.get_relevant_documents(query)
        if hasattr(_rag_retriever, "search"):
            return _rag_retriever.search(query, top_k=top_k)  # type: ignore
        if hasattr(_rag_retriever, "get_relevant_documents"):
            return _rag_retriever.get_relevant_documents(query)[:top_k]  # type: ignore

        raise RuntimeError("Retriever does not support search/get_relevant_documents")

    def _format_results(self, results: Sequence[Any], sources: List[str]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for r in results:
            text, source, doc_id, score, meta = self._extract_result_fields(r)
            text = self._clean_text(text)

            if not text:
                continue

            chunks.append(
                {
                    "text": text,
                    "source": source,
                    "doc_id": doc_id,
                    "score": score,
                    "metadata": meta,
                }
            )

        # (Optional) source filtering could happen here if your retriever spans multiple corpora.
        # For v1, we keep all results and trust retrieval metadata.

        return chunks

    def _extract_result_fields(self, r: Any) -> Tuple[str, Optional[str], Optional[str], Optional[float], Optional[Dict[str, Any]]]:
        """
        Best-effort extraction from common result shapes:
        - result.chunk.text / result.chunk.metadata
        - result.page_content / result.metadata (LangChain docs)
        - dict with keys
        """
        text = ""
        source = None
        doc_id = None
        score = None
        meta = None

        # dict-like
        if isinstance(r, dict):
            text = r.get("text") or r.get("content") or ""
            source = r.get("source")
            doc_id = r.get("doc_id")
            score = r.get("score")
            meta = r.get("metadata")
            return str(text), source, doc_id, score if isinstance(score, (int, float)) else None, meta

        # LangChain Document-like
        if hasattr(r, "page_content"):
            text = getattr(r, "page_content", "") or ""
            meta = getattr(r, "metadata", None)
            if isinstance(meta, dict):
                source = meta.get("source") or meta.get("file") or meta.get("path")
                doc_id = meta.get("doc_id") or meta.get("id")
            return text, source, doc_id, None, meta if isinstance(meta, dict) else None

        # Custom result with .chunk
        if hasattr(r, "chunk"):
            ch = getattr(r, "chunk")
            if hasattr(ch, "text"):
                text = getattr(ch, "text", "") or ""
            else:
                text = str(ch)

            # metadata
            meta = getattr(ch, "metadata", None)
            if isinstance(meta, dict):
                source = meta.get("source") or meta.get("file") or meta.get("path")
                doc_id = meta.get("doc_id") or meta.get("id")

            score_val = getattr(r, "score", None)
            if isinstance(score_val, (int, float)):
                score = float(score_val)

            return text, source, doc_id, score, meta if isinstance(meta, dict) else None

        # fallback
        return str(r), None, None, None, None

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 1200:
            text = text[:1200] + "..."
        return text


# ---------------------------------------------------------------------
# Helper functions for other agents
# ---------------------------------------------------------------------

def create_rag_request(
    query: str,
    sources: Optional[List[str]] = None,
    context_type: str = "general",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Standard helper used by other agents.
    """
    return {
        "query": query,
        "sources": sources or RagDataSource.ALL,
        "context_type": context_type,
        "top_k": top_k,
    }
