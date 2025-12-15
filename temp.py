from __future__ import annotations

from typing import Any, List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agentic_logging import get_logger
from agentic_suite.graph.state import GraphState
from agentic_suite.llm.prompt_loader import load_prompt

logger = get_logger("chat_agent")


class ChatAgent:
    """
    Chat agent node for LangGraph.

    Contract:
    - Input state["messages"] may contain:
        - LangChain BaseMessage objects (preferred)
        - (optionally) legacy dict messages {role, content, ...} for backward compatibility
    - Output state["messages"] WILL contain LangChain BaseMessage objects.

    Tool calling:
    - The LLM may emit tool_calls in the returned AIMessage
    - LangGraph routing (tools_condition) should send control to ToolNode
    - ToolNode appends ToolMessage(s) and routes back to this ChatAgent
    - This ChatAgent then generates the final natural-language answer

    RAG:
    - If state["rag_status"] == "done", ChatAgent will inject RAG context as a SystemMessage once.
    - If enable_rag_autorequest is True, ChatAgent may set:
        state["rag_status"] = "requested"
        state["rag_request"] = {...}
    """

    def __init__(
        self,
        llm: Any,
        *,
        max_rag_iterations: int = 3,
        enable_rag_autorequest: bool = True,
        rag_injection_max_chunks: int = 3,
        rag_injection_max_chars: int = 600,
    ):
        self.llm = llm
        self.max_rag_iterations = max_rag_iterations
        self.enable_rag_autorequest = enable_rag_autorequest
        self.rag_injection_max_chunks = rag_injection_max_chunks
        self.rag_injection_max_chars = rag_injection_max_chars

    def __call__(self, state: GraphState) -> GraphState:
        session_id = state.get("session_id", "unknown")
        logger.info("ChatAgent invoked", extra={"session_id": session_id})

        # 1) Normalize incoming messages to LangChain BaseMessage objects
        messages: List[BaseMessage] = self._normalize_messages(state.get("messages") or [])

        # 2) If RAG just finished, inject context ONCE, then clear rag flags
        if state.get("rag_status") == "done":
            rag_context = self._build_rag_context(state.get("rag_result"))
            if rag_context:
                messages.append(SystemMessage(content=rag_context))

            # Clear so we don't inject repeatedly
            state["rag_status"] = "idle"
            state["rag_result"] = None
            state["rag_request"] = None

        # 3) Load system prompt (prepend each invocation, simplest + safe)
        system_prompt = self._load_system_prompt()
        full_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)] + messages

        # 4) Call the LLM
        try:
            response = self.llm.invoke(full_messages)
        except Exception as e:
            logger.error(
                "ChatAgent LLM invoke failed",
                extra={"session_id": session_id, "error": str(e)},
            )
            response = AIMessage(
                content="I hit an error while generating a response. Please try again."
            )

        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))

        # 5) Append assistant response
        messages.append(response)

        # 6) Save draft for API gateway
        state["llm_response_draft"] = response.content or ""

        # 7) Store LangChain messages back into state (preferred format)
        state["messages"] = messages

        # 8) Optionally request RAG (light heuristic). Safe to disable.
        if self.enable_rag_autorequest:
            self._maybe_request_rag(state)

        return state

    # -----------------------
    # Internals
    # -----------------------

    def _load_system_prompt(self) -> str:
        try:
            return load_prompt("chat")
        except FileNotFoundError:
            logger.warning("Prompt 'chat' not found, using default system prompt")
            return (
                "You are the Agentic DL Workflow Suite assistant.\n"
                "- You may call tools when needed (devices, runs, conversions, profiling).\n"
                "- If you are unsure about tool flags, framework steps, or docs, request RAG.\n"
                "- Be concise and action-oriented."
            )

    def _normalize_messages(self, raw: Sequence[Any]) -> List[BaseMessage]:
        """
        Convert mixed message formats into LangChain BaseMessage objects.
        Supports legacy dict messages for smooth migration.
        """
        out: List[BaseMessage] = []
        for m in raw:
            if isinstance(m, BaseMessage):
                out.append(m)
                continue

            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content", "") or ""

                if role == "system":
                    out.append(SystemMessage(content=content))
                elif role == "user":
                    out.append(HumanMessage(content=content))
                elif role == "assistant":
                    tool_calls = m.get("tool_calls") or None
                    if tool_calls:
                        # Preserve tool_calls if your upstream stored them
                        out.append(AIMessage(content=content, tool_calls=tool_calls))
                    else:
                        out.append(AIMessage(content=content))
                elif role == "tool":
                    # ToolMessage ideally includes tool_call_id; keep best-effort
                    tool_call_id = m.get("tool_call_id") or "unknown"
                    out.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                else:
                    out.append(AIMessage(content=content))
                continue

            # Unknown type -> stringify
            out.append(AIMessage(content=str(m)))

        return out

    def _build_rag_context(self, rag_result: Optional[Any]) -> str:
        """
        Accepts rag_result in dict or object form. Expected shapes:
          - {"chunks": [{"text": "...", "source": "..."} , ...]}
          - RagResult with .chunks where each chunk has .text/.source or dict-like
        Returns a compact SystemMessage content string.
        """
        if rag_result is None:
            return ""

        chunks = None
        if isinstance(rag_result, dict):
            chunks = rag_result.get("chunks")
        else:
            chunks = getattr(rag_result, "chunks", None)

        if not chunks:
            return ""

        lines: List[str] = ["Relevant context from local documentation:"]
        for i, ch in enumerate(chunks[: self.rag_injection_max_chunks]):
            text, source = "", ""

            if isinstance(ch, dict):
                text = (ch.get("text") or ch.get("chunk") or "")[: self.rag_injection_max_chars]
                source = ch.get("source") or ch.get("doc_id") or ""
            else:
                text = (getattr(ch, "text", "") or "")[: self.rag_injection_max_chars]
                source = getattr(ch, "source", "") or ""

            if source:
                lines.append(f"{i+1}. [{source}] {text}")
            else:
                lines.append(f"{i+1}. {text}")

        return "\n".join(lines)

    def _maybe_request_rag(self, state: GraphState) -> None:
        """
        Minimal heuristic:
        - If the assistant expresses uncertainty, request RAG
        - Uses last user message as query when possible
        """
        # Do not request if already requested/in progress
        if state.get("rag_status") in ("requested", "running"):
            return

        rag_iterations = int(state.get("rag_iterations") or 0)
        if rag_iterations >= self.max_rag_iterations:
            return

        messages: List[BaseMessage] = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            return

        assistant_text = (messages[-1].content or "").lower()

        triggers = (
            "i'm not sure",
            "not certain",
            "need to check",
            "depends on the version",
            "refer to documentation",
            "check the docs",
            "i can't confirm",
            "unknown flag",
        )

        if not any(t in assistant_text for t in triggers):
            return

        # Try to get the last user message as the retrieval query
        last_user = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user = m.content
                break

        query = (last_user or "").strip()
        if not query:
            query = "Find relevant documentation for the current question."

        state["rag_request"] = {
            "query": query,
            "max_chunks": 3,
            "context_type": "general",
        }
        state["rag_status"] = "requested"
        state["rag_iterations"] = rag_iterations + 1

        logger.info(
            "ChatAgent requested RAG",
            extra={"session_id": state.get("session_id", "unknown"), "query": query},
        )
