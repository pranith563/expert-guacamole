# apps/orchestrator/agentic_suite/graph/state.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from langchain_core.messages import BaseMessage


# -----------------------
# Mode / Task literals
# -----------------------

Mode = Literal["chat", "convert", "profile", "patch_search"]

TaskType = Literal[
    "chat",
    "convert",
    "profile",
    "patch_search",
    "analysis",
]


# -----------------------
# UI Context
# -----------------------

class UIContext(TypedDict, total=False):
    """
    UI context about what the user is viewing / has selected.
    Keep this lightweight and stable; agents use this to ground actions.
    """
    selected_project_id: Optional[str]
    selected_model_id: Optional[str]
    selected_artifact_id: Optional[str]
    selected_experiment_id: Optional[str]

    active_pane: Optional[Literal["left", "right"]]
    visible_files: List[str]
    cursor_position: Optional[Dict[str, Any]]  # e.g. {"file": "...", "line": 12, "col": 4}


# -----------------------
# Task parameter blocks
# -----------------------

class ConvertParams(TypedDict, total=False):
    input_model_path: str
    input_framework: Literal["tensorflow", "pytorch", "onnx"]
    target_runtime: Literal["tflite", "qnn", "snpe", "neuropilot"]
    target_accelerator: Literal["cpu", "gpu", "htp", "mdla", "dsp"]
    quantization: Optional[Literal["none", "int8", "fp16"]]
    optimization_level: int  # e.g. 0-3
    custom_options: Dict[str, Any]


class ProfileParams(TypedDict, total=False):
    model_path: str
    device_id: Optional[str]
    accelerator: Literal["cpu", "gpu", "htp", "mdla", "dsp"]
    iterations: int
    warmup_iterations: int
    collect_memory: bool
    collect_power: bool
    custom_options: Dict[str, Any]


class PatchSearchParams(TypedDict, total=False):
    query: str
    error_message: Optional[str]
    context_model: Optional[str]
    context_runtime: Optional[str]
    max_results: int
    sources: List[Literal["docs", "confluence", "experiments", "modelhub"]]


class AnalysisParams(TypedDict, total=False):
    experiment_ids: List[str]
    analysis_type: Literal["single", "comparison", "trend"]
    metrics: List[str]  # ["latency", "memory", "power", ...]
    comparison_baseline: Optional[str]


# -----------------------
# Runtime tracking
# -----------------------

class IntermediateResult(TypedDict, total=False):
    step_name: str
    step_index: int
    data: Any
    metadata: Dict[str, Any]


class LLMPlan(TypedDict, total=False):
    goal: str
    steps: List[str]
    current_step: int
    reasoning: str


# -----------------------
# Legacy message dict (optional support)
# -----------------------

class MessageDict(TypedDict, total=False):
    """
    Legacy/compat message format. Prefer LangChain BaseMessage objects in state["messages"].
    """
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str]
    tool_call_id: Optional[str]
    tool_calls: Any
    metadata: Dict[str, Any]


MessageLike = Union[BaseMessage, MessageDict]


# -----------------------
# RAG dataclasses (kept small + serializable)
# -----------------------

@dataclass
class RagRequest:
    query: str
    max_chunks: Optional[int] = None
    sources: Optional[List[str]] = None
    context_type: str = "general"


@dataclass
class RagChunk:
    text: str
    source: Optional[str] = None
    doc_id: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RagResult:
    used_query: str
    chunks: List[RagChunk]
    notes: Optional[str] = None
    sources_used: Optional[List[str]] = None


RagStatus = Literal["idle", "requested", "running", "done", "failed"]


# -----------------------
# Main GraphState
# -----------------------

class GraphState(TypedDict, total=False):
    """
    Complete session state flowing through LangGraph.
    Keep keys consistent across API gateway, supervisor, agents, and graph wiring.
    """

    # Core
    user_id: str
    session_id: str
    mode: Mode
    ui_context: UIContext
    messages: List[MessageLike]

    # Intent / task
    intent: Optional[str]
    sub_intent: Optional[str]
    active_task_type: Optional[TaskType]
    active_task_id: Optional[str]

    # Params
    convert_params: Optional[ConvertParams]
    profile_params: Optional[ProfileParams]
    patch_search_params: Optional[PatchSearchParams]
    analysis_params: Optional[AnalysisParams]

    # Experiments / tools
    experiment_run_id: Optional[str]
    last_error: Optional[str]
    intermediate_results: List[IntermediateResult]

    # LLM planning / response
    llm_plan: Optional[LLMPlan]
    llm_response_draft: Optional[str]

    # RAG
    rag_request: Optional[Union[RagRequest, Dict[str, Any]]]
    rag_result: Optional[Union[RagResult, Dict[str, Any]]]
    rag_status: RagStatus
    rag_observations: List[str]
    rag_iterations: int
    max_rag_iterations: int


# -----------------------
# State factory helpers
# -----------------------

def create_initial_state(
    *,
    user_id: str,
    session_id: str,
    mode: Mode = "chat",
    max_rag_iterations: int = 3,
) -> GraphState:
    """
    Create a new initial GraphState with stable defaults.
    Messages are initialized empty; agents will append LangChain BaseMessage objects.
    """
    return {
        # Core
        "user_id": user_id,
        "session_id": session_id,
        "mode": mode,
        "ui_context": {
            "selected_project_id": None,
            "selected_model_id": None,
            "selected_artifact_id": None,
            "selected_experiment_id": None,
        },
        "messages": [],

        # Intent / task
        "intent": None,
        "sub_intent": None,
        "active_task_type": None,
        "active_task_id": None,

        # Params
        "convert_params": None,
        "profile_params": None,
        "patch_search_params": None,
        "analysis_params": None,

        # Runtime
        "experiment_run_id": None,
        "last_error": None,
        "intermediate_results": [],

        # LLM
        "llm_plan": None,
        "llm_response_draft": None,

        # RAG
        "rag_request": None,
        "rag_result": None,
        "rag_status": "idle",
        "rag_observations": [],
        "rag_iterations": 0,
        "max_rag_iterations": max_rag_iterations,
    }


def reset_task_state(state: GraphState) -> GraphState:
    """
    Reset task-specific fields while preserving core session context + message history.
    Useful when starting a new task within the same thread/session.
    """
    # Preserve core fields
    preserved: GraphState = {
        "user_id": state.get("user_id", ""),
        "session_id": state.get("session_id", ""),
        "mode": state.get("mode", "chat"),  # keep current mode
        "ui_context": state.get("ui_context", {}),
        "messages": state.get("messages", []),
        "max_rag_iterations": state.get("max_rag_iterations", 3),
    }

    # Reset task fields
    preserved.update(
        {
            "intent": None,
            "sub_intent": None,
            "active_task_type": None,
            "active_task_id": None,

            "convert_params": None,
            "profile_params": None,
            "patch_search_params": None,
            "analysis_params": None,

            "experiment_run_id": None,
            "last_error": None,
            "intermediate_results": [],

            "llm_plan": None,
            "llm_response_draft": None,

            "rag_request": None,
            "rag_result": None,
            "rag_status": "idle",
            "rag_observations": [],
            "rag_iterations": 0,
        }
    )
    return preserved
