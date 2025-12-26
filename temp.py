
# agents/converter/__init__.py

"""
ConverterAgent sub-graph implementation.

This package contains:
- Typed schemas (intent -> plan -> result)
- Dependency facades (ModelHub + SNPE/QNN MCP reflective tools)
- LangGraph sub-graph (deterministic execution)
"""

from .schema import ConversionIntent, ConversionResult
from .graph import build_converter_subgraph

# agents/converter/schema.py

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelRefType(str, Enum):
    modelhub_id = "modelhub_id"
    path = "path"


class ModelRef(BaseModel):
    """
    Reference to a model either by ModelHub id or by repository path.
    Note: absolute paths should come ONLY from ModelHub resolution,
    not from user input (unless you explicitly allow it).
    """
    type: ModelRefType
    value: str


class ConversionIntent(BaseModel):
    """
    The ONLY object the system should reason over.
    No CLI flags. No SDK internals.
    """
    model_ref: ModelRef
    source_framework: str
    target_runtime: str
    target_soc: str

    precision: str = Field(default="fp16")           # fp32|fp16|int8 etc
    quantization: str = Field(default="none")        # none|static|dynamic

    user_constraints: Optional[Dict[str, str]] = None


class WorkspacePaths(BaseModel):
    """
    Absolute workspace locations for this experiment run.
    """
    root: str
    input_dir: str
    converted_dir: str
    logs_dir: str


class ResolvedToolPlan(BaseModel):
    """
    Fully resolved, deterministic execution plan.
    """
    tool_name: str
    tool_category: str
    required_args: Dict[str, Any]
    optional_args: Dict[str, Any]
    workspace: WorkspacePaths


class ConversionResult(BaseModel):
    """
    Output that gets copied back into the main GraphState.
    Keep it small and stable (UI/API-friendly).
    """
    experiment_id: str
    status: str
    artifacts: List[Dict[str, str]] = Field(default_factory=list)
    error: Optional[str] = None

# agents/converter/state.py

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import MessagesState

from .schema import ConversionIntent, ResolvedToolPlan, WorkspacePaths, ConversionResult


class ConverterState(MessagesState):
    """
    Converter sub-graph state.
    This is intentionally separate from the main orchestrator GraphState.

    Only pass in what you want Converter to see (typically ConversionIntent + session/thread id).
    """

    # Provided by caller
    intent: Optional[ConversionIntent] = None
    user: Optional[str] = None

    # Resolved model metadata (from ModelHub)
    model: Optional[Dict[str, Any]] = None

    # Experiment tracking
    experiment_id: Optional[str] = None
    workspace: Optional[WorkspacePaths] = None

    # Tooling
    selected_tool: Optional[str] = None
    tool_plan: Optional[ResolvedToolPlan] = None

    # Execution bookkeeping
    failed: bool = False
    error: Optional[str] = None

    # Final compact output
    result: Optional[ConversionResult] = None

	# agents/converter/deps.py

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable, Optional


@runtime_checkable
class AsyncTool(Protocol):
    """
    Minimal protocol for LangChain tools that support async invocation.
    MCP tools returned by MultiServerMCPClient.get_tools() typically support `ainvoke`.
    """
    name: str

    async def ainvoke(self, input: Dict[str, Any]) -> Any: ...


class ModelHubClient(Protocol):
    async def get_model(self, model_id: str) -> Dict[str, Any]: ...
    async def get_by_path(self, file_path: str) -> Dict[str, Any]: ...


class SnpeQnnRegistryClient(Protocol):
    async def list_tools(self) -> List[Dict[str, Any]]: ...
    async def get_tool_spec(self, tool_name: str, view: str) -> Dict[str, Any]: ...
    async def run_tool(self, tool_name: str, args: Dict[str, Any], workspace_root: str) -> Dict[str, Any]: ...


class ExperimentService(Protocol):
    """
    Replace with your DB-backed implementation.
    """
    def create(
        self,
        experiment_id: str,
        user: str,
        model_id: str,
        target_runtime: str,
        target_soc: str,
        status: str,
        created_at: Any,
        metadata: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def update_status(
        self,
        experiment_id: str,
        status: str,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def add_artifacts(self, experiment_id: str, artifacts: List[Dict[str, str]]) -> None: ...


class ModelHubMcpFacade(ModelHubClient):
    """
    Wrap ModelHub MCP tools into a small typed client.
    """
    def __init__(self, tool_get_model: AsyncTool, tool_get_by_path: AsyncTool):
        self._get_model = tool_get_model
        self._get_by_path = tool_get_by_path

    async def get_model(self, model_id: str) -> Dict[str, Any]:
        return await self._get_model.ainvoke({"model_id": model_id})

    async def get_by_path(self, file_path: str) -> Dict[str, Any]:
        return await self._get_by_path.ainvoke({"file_path": file_path})


class SnpeQnnMcpFacade(SnpeQnnRegistryClient):
    """
    Wrap SNPE/QNN reflective MCP tools:
      - list_tools
      - get_tool_spec
      - run_tool
    """
    def __init__(self, tool_list_tools: AsyncTool, tool_get_tool_spec: AsyncTool, tool_run_tool: AsyncTool):
        self._list_tools = tool_list_tools
        self._get_tool_spec = tool_get_tool_spec
        self._run_tool = tool_run_tool

    async def list_tools(self) -> List[Dict[str, Any]]:
        return await self._list_tools.ainvoke({})

    async def get_tool_spec(self, tool_name: str, view: str) -> Dict[str, Any]:
        return await self._get_tool_spec.ainvoke({"tool_name": tool_name, "view": view})

    async def run_tool(self, tool_name: str, args: Dict[str, Any], workspace_root: str) -> Dict[str, Any]:
        return await self._run_tool.ainvoke(
            {"tool_name": tool_name, "args": args, "workspace_root": workspace_root}
        )

	# agents/converter/nodes.py

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .schema import ConversionResult, ResolvedToolPlan, WorkspacePaths
from .state import ConverterState
from .deps import ModelHubClient, SnpeQnnRegistryClient, ExperimentService


# -----------------------
# Helpers
# -----------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _relpath(workspace_root: str, abs_path: str) -> str:
    return os.path.relpath(abs_path, workspace_root)


def _pick_single_tool(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not matches:
        raise RuntimeError("No compatible conversion tool found")
    if len(matches) > 1:
        names = [m.get("name", "<unnamed>") for m in matches]
        raise RuntimeError(f"Ambiguous conversion tools: {names}")
    return matches[0]


def _infer_required_arg_value(
    arg_name: str,
    input_model_abs: str,
    output_dir_abs: str,
) -> Optional[str]:
    """
    Conservative mapping of required args.
    Adjust this mapping to your tool spec naming conventions.

    IMPORTANT: Do not guess exotic flags/args here. Only map obvious I/O arguments.
    """
    n = arg_name.lower()

    # input model
    if ("input" in n and "model" in n) or n in ("input_model", "model", "input"):
        return input_model_abs

    # output artifact
    if ("output" in n and ("dlc" in n or "model" in n)) or n in ("output_dlc", "output_model", "out"):
        # Default DLC name. If your tool spec uses different output type, adjust.
        return os.path.join(output_dir_abs, "model.dlc")

    return None


def _validate_optional_args(optional: Dict[str, Any], optional_spec: Dict[str, Any]) -> None:
    """
    Optional: validate optional arg names exist in spec to catch typos/policy drift.
    This avoids silently passing unknown params into run_tool.
    """
    spec_args = optional_spec.get("args") or []
    allowed = {a.get("name") for a in spec_args if a.get("name")}
    unknown = [k for k in optional.keys() if k not in allowed]
    if unknown:
        raise RuntimeError(f"Unknown optional args for tool: {unknown}")


# -----------------------
# Nodes
# -----------------------

async def resolve_model(state: ConverterState, modelhub: ModelHubClient) -> ConverterState:
    """
    Resolve model metadata and absolute path via ModelHub (read-only).
    Expected model dict keys (adjust if your ModelHub MCP differs):
      - id
      - filename
      - absolute_path
    """
    intent = state.intent
    if intent is None:
        raise RuntimeError("ConversionIntent missing")

    if intent.model_ref.type.value == "modelhub_id":
        model = await modelhub.get_model(intent.model_ref.value)
    else:
        model = await modelhub.get_by_path(intent.model_ref.value)

    # Basic validation
    for k in ("id", "filename", "absolute_path"):
        if k not in model:
            raise RuntimeError(f"ModelHub response missing '{k}'. Got keys: {list(model.keys())}")

    state.model = model
    return state


async def create_experiment(
    state: ConverterState,
    experiments: ExperimentService,
) -> ConverterState:
    """
    Create an experiment DB record early so we can track failures.
    """
    if not state.user:
        raise RuntimeError("state.user missing")
    if not state.model:
        raise RuntimeError("state.model missing")

    # You can swap this to uuid4 if you prefer — keep stable format.
    import uuid
    exp_id = str(uuid.uuid4())

    experiments.create(
        experiment_id=exp_id,
        user=state.user,
        model_id=state.model["id"],
        target_runtime=state.intent.target_runtime,  # type: ignore[union-attr]
        target_soc=state.intent.target_soc,          # type: ignore[union-attr]
        status="CREATED",
        created_at=datetime.utcnow(),
        metadata={"source_framework": state.intent.source_framework, "precision": state.intent.precision},  # type: ignore[union-attr]
        params=state.intent.model_dump(),  # type: ignore[union-attr]
    )

    state.experiment_id = exp_id
    return state


async def materialize_workspace(
    state: ConverterState,
    experiments: ExperimentService,
    experiments_root: str = "/data/experiments",
) -> ConverterState:
    """
    Create workspace directories and copy model artifacts into input/.
    Write metadata.json and params.json for reproducibility.
    """
    if not state.experiment_id or not state.model or not state.user:
        raise RuntimeError("Missing experiment_id/model/user")

    ws_root = os.path.join(experiments_root, state.user, state.experiment_id)
    input_dir = os.path.join(ws_root, "input")
    converted_dir = os.path.join(ws_root, "converted")
    logs_dir = os.path.join(ws_root, "logs")

    _ensure_dir(input_dir)
    _ensure_dir(converted_dir)
    _ensure_dir(logs_dir)

    # Copy model into workspace input/
    src = state.model["absolute_path"]
    dst = os.path.join(input_dir, state.model["filename"])
    shutil.copy2(src, dst)

    # Write metadata & params (small, stable)
    _write_json(os.path.join(ws_root, "metadata.json"), {
        "experiment_id": state.experiment_id,
        "user": state.user,
        "model_id": state.model["id"],
        "model_filename": state.model["filename"],
        "source_model_copied_from": src,  # If you consider this sensitive, remove or store relative.
        "target_runtime": state.intent.target_runtime,  # type: ignore[union-attr]
        "target_soc": state.intent.target_soc,          # type: ignore[union-attr]
    })
    _write_json(os.path.join(ws_root, "params.json"), state.intent.model_dump())  # type: ignore[union-attr]

    state.workspace = WorkspacePaths(
        root=ws_root,
        input_dir=input_dir,
        converted_dir=converted_dir,
        logs_dir=logs_dir,
    )

    experiments.update_status(state.experiment_id, "MATERIALIZED")
    return state


async def select_tool(state: ConverterState, snpe: SnpeQnnRegistryClient) -> ConverterState:
    """
    Deterministically pick a single matching conversion tool based on intent.
    """
    if not state.intent:
        raise RuntimeError("intent missing")

    tools = await snpe.list_tools()

    matches = [
        t for t in tools
        if t.get("category") == "conversion"
        and t.get("source") == state.intent.source_framework
        and t.get("target") == state.intent.target_runtime
    ]

    picked = _pick_single_tool(matches)
    state.selected_tool = picked["name"]
    return state


async def resolve_tool_plan(state: ConverterState, snpe: SnpeQnnRegistryClient) -> ConverterState:
    """
    Fetch required args spec and resolve only obvious required args (input/output).
    Optional args are applied only by explicit intent/policy.
    """
    if not state.selected_tool or not state.workspace or not state.model or not state.intent:
        raise RuntimeError("Missing selected_tool/workspace/model/intent")

    tool_name = state.selected_tool
    required_spec = await snpe.get_tool_spec(tool_name=tool_name, view="required")

    required_args: Dict[str, Any] = {}
    spec_args = required_spec.get("args") or []
    if not spec_args:
        raise RuntimeError(f"Tool '{tool_name}' returned empty required spec")

    input_model_abs = os.path.join(state.workspace.input_dir, state.model["filename"])
    output_dir_abs = state.workspace.converted_dir

    for arg in spec_args:
        arg_name = arg.get("name")
        if not arg_name:
            continue

        inferred = _infer_required_arg_value(arg_name, input_model_abs, output_dir_abs)
        if inferred is None:
            # We refuse to guess non-I/O required args. Force explicit mapping.
            raise RuntimeError(
                f"Unhandled required arg '{arg_name}' for tool '{tool_name}'. "
                "Add mapping in _infer_required_arg_value() or supply via intent/policy."
            )
        required_args[arg_name] = inferred

    # Optional args from intent/policy (minimal)
    optional_args: Dict[str, Any] = {}
    if state.intent.precision:
        optional_args["precision"] = state.intent.precision
    if state.intent.quantization and state.intent.quantization != "none":
        optional_args["quantization"] = state.intent.quantization

    # Optional validation (recommended)
    if optional_args:
        optional_spec = await snpe.get_tool_spec(tool_name=tool_name, view="optional")
        _validate_optional_args(optional_args, optional_spec)

    state.tool_plan = ResolvedToolPlan(
        tool_name=tool_name,
        tool_category="conversion",
        required_args=required_args,
        optional_args=optional_args,
        workspace=state.workspace,
    )
    return state


async def run_conversion(
    state: ConverterState,
    snpe: SnpeQnnRegistryClient,
    experiments: ExperimentService,
) -> ConverterState:
    """
    Execute the conversion via MCP `run_tool`.
    Capture logs and update artifacts.
    """
    if not state.experiment_id or not state.tool_plan or not state.workspace:
        raise RuntimeError("Missing experiment_id/tool_plan/workspace")

    experiments.update_status(state.experiment_id, "CONVERTING")

    # Run tool
    exec_result = await snpe.run_tool(
        tool_name=state.tool_plan.tool_name,
        args={**state.tool_plan.required_args, **state.tool_plan.optional_args},
        workspace_root=state.workspace.root,
    )

    # Persist logs (best-effort)
    stdout = str(exec_result.get("stdout", ""))
    stderr = str(exec_result.get("stderr", ""))
    exit_code = exec_result.get("exit_code")

    _write_text(os.path.join(state.workspace.logs_dir, "conversion.stdout.txt"), stdout)
    _write_text(os.path.join(state.workspace.logs_dir, "conversion.stderr.txt"), stderr)
    _write_json(os.path.join(state.workspace.logs_dir, "conversion.exec_result.json"), exec_result)

    if exit_code not in (0, "0", None):
        raise RuntimeError(f"Conversion tool failed (exit_code={exit_code}). See logs/")

    # Verify primary artifact exists (default DLC)
    # If your output arg name differs, adjust this.
    output_candidates = [
        v for k, v in state.tool_plan.required_args.items()
        if "output" in k.lower()
    ]
    if not output_candidates:
        raise RuntimeError("No output path resolved from required args")

    primary_out = output_candidates[0]
    if not os.path.exists(primary_out):
        raise RuntimeError(f"Expected output artifact missing: {primary_out}")

    artifacts = [{"type": "dlc", "path": _relpath(state.workspace.root, primary_out)}]
    experiments.add_artifacts(state.experiment_id, artifacts)

    experiments.update_status(state.experiment_id, "COMPLETED")

    state.result = ConversionResult(
        experiment_id=state.experiment_id,
        status="COMPLETED",
        artifacts=artifacts,
    )
    return state


async def handle_failure(state: ConverterState, experiments: ExperimentService) -> ConverterState:
    """
    Single sink for structured failures:
    - mark experiment FAILED
    - write error log
    - produce compact ConversionResult with error
    """
    err = state.error or "Unknown conversion failure"

    if state.workspace:
        try:
            _write_text(os.path.join(state.workspace.logs_dir, "failure.txt"), err)
        except Exception:
            pass

    if state.experiment_id:
        experiments.update_status(state.experiment_id, "FAILED", error=err)

    state.result = ConversionResult(
        experiment_id=state.experiment_id or "<missing>",
        status="FAILED",
        artifacts=[],
        error=err,
    )
    return state

# agents/converter/graph.py

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langgraph.graph import StateGraph, END

from .state import ConverterState
from .deps import ModelHubClient, SnpeQnnRegistryClient, ExperimentService
from . import nodes


AsyncNode = Callable[[ConverterState], Awaitable[ConverterState]]


def _safe(node_name: str, fn: AsyncNode) -> AsyncNode:
    """
    Wrap any node to prevent bubbling exceptions out of the subgraph.
    We record failure in state and rely on conditional edges to route to handle_failure.
    """
    async def _wrapped(state: ConverterState) -> ConverterState:
        try:
            return await fn(state)
        except Exception as e:
            state.failed = True
            state.error = f"[{node_name}] {e}"
            return state
    return _wrapped


def _route_ok_fail(state: ConverterState) -> str:
    return "fail" if state.failed else "ok"


def build_converter_subgraph(
    *,
    modelhub: ModelHubClient,
    snpe: SnpeQnnRegistryClient,
    experiments: ExperimentService,
    experiments_root: str = "/data/experiments",
):
    """
    Build and compile the Converter sub-graph.
    """

    g = StateGraph(ConverterState)

    # Nodes (safe-wrapped)
    g.add_node("resolve_model", _safe("resolve_model", lambda s: nodes.resolve_model(s, modelhub)))
    g.add_node("create_experiment", _safe("create_experiment", lambda s: nodes.create_experiment(s, experiments)))
    g.add_node("materialize_workspace", _safe(
        "materialize_workspace",
        lambda s: nodes.materialize_workspace(s, experiments, experiments_root=experiments_root),
    ))
    g.add_node("select_tool", _safe("select_tool", lambda s: nodes.select_tool(s, snpe)))
    g.add_node("resolve_tool_plan", _safe("resolve_tool_plan", lambda s: nodes.resolve_tool_plan(s, snpe)))
    g.add_node("run_conversion", _safe("run_conversion", lambda s: nodes.run_conversion(s, snpe, experiments)))
    g.add_node("handle_failure", lambda s: nodes.handle_failure(s, experiments))

    # Entry
    g.set_entry_point("resolve_model")

    # Conditional flow after each step
    g.add_conditional_edges("resolve_model", _route_ok_fail, {"ok": "create_experiment", "fail": "handle_failure"})
    g.add_conditional_edges("create_experiment", _route_ok_fail, {"ok": "materialize_workspace", "fail": "handle_failure"})
    g.add_conditional_edges("materialize_workspace", _route_ok_fail, {"ok": "select_tool", "fail": "handle_failure"})
    g.add_conditional_edges("select_tool", _route_ok_fail, {"ok": "resolve_tool_plan", "fail": "handle_failure"})
    g.add_conditional_edges("resolve_tool_plan", _route_ok_fail, {"ok": "run_conversion", "fail": "handle_failure"})

    # Terminal edges
    g.add_edge("run_conversion", END)
    g.add_edge("handle_failure", END)

    return g.compile()

# agentic_suite/agents/converter_agent.py

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.converter.schema import ConversionIntent, ConversionResult
from agents.converter.graph import build_converter_subgraph


class ConverterAgent:
    """
    Main-graph node wrapper around the Converter sub-graph.

    Input (from main GraphState):
      - state["conversion_intent"] : dict | ConversionIntent
      - state["session_id"]        : str (optional but recommended)

    Output (into main GraphState):
      - state["conversion_result"] : dict
      - state["last_experiment_id"]: str
    """

    def __init__(
        self,
        *,
        modelhub_client: Any,
        snpe_client: Any,
        experiment_service: Any,
        experiments_root: str = "/data/experiments",
        user_provider: Optional[Any] = None,
    ):
        self._experiments_root = experiments_root
        self._user_provider = user_provider

        # Compile subgraph once
        self._subgraph = build_converter_subgraph(
            modelhub=modelhub_client,
            snpe=snpe_client,
            experiments=experiment_service,
            experiments_root=experiments_root,
        )

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raw_intent = state.get("conversion_intent")
        if raw_intent is None:
            raise RuntimeError("conversion_intent missing in main state")

        # Determine user (from auth/session or fallback)
        user = None
        if self._user_provider is not None:
            user = self._user_provider(state)
        if not user:
            user = state.get("user") or "default_user"

        # Parse intent
        intent = raw_intent if isinstance(raw_intent, ConversionIntent) else ConversionIntent.model_validate(raw_intent)

        # Use main session id for continuity/checkpointing if you want
        thread_id = state.get("session_id") or state.get("thread_id") or "converter-session"

        sub_state = {
            "messages": [],   # Converter is non-chat; keep empty
            "intent": intent,
            "user": user,
        }

        out = await self._subgraph.ainvoke(sub_state, config={"thread_id": thread_id})

        result: ConversionResult = out.get("result")
        if result is None:
            raise RuntimeError("Converter subgraph produced no result")

        # Copy compact output back to main graph state
        state["conversion_result"] = result.model_dump()
        state["last_experiment_id"] = result.experiment_id

        # Optional: set a user-facing message for ChatAgent to summarize later
        # (ONLY if your ChatAgent reads this field)
        state["last_task_summary"] = {
            "task": "convert",
            "status": result.status,
            "experiment_id": result.experiment_id,
        }

        return state

# ConverterAgent

ConverterAgent is a **deterministic execution sub-graph** responsible for converting deep learning models
into target runtimes (SNPE/QNN/etc.) inside the Agentic DL Workflow Suite.

It is intentionally **non-conversational**, **reproducible**, and **auditable**.

---

## Purpose

ConverterAgent performs:

- Resolve a model (ModelHub is read-only)
- Create an experiment DB record
- Materialize an isolated workspace
- Select the correct conversion tool via MCP (reflective registry)
- Resolve required args deterministically (I/O only)
- Apply optional args only from intent/policy
- Execute conversion via MCP `run_tool`
- Persist logs and artifacts
- Mark experiment `COMPLETED` or `FAILED`

---

## Inputs

ConverterAgent consumes a `ConversionIntent`:

```json
{
  "model_ref": {"type": "modelhub_id|path", "value": "string"},
  "source_framework": "tensorflow|pytorch|onnx|tflite",
  "target_runtime": "snpe|qnn|tflite|mdla",
  "target_soc": "string",
  "precision": "fp32|fp16|int8",
  "quantization": "none|static|dynamic"
}

# agents/converter/prompts.py

CONVERTER_SYSTEM_PROMPT = """\
You are ConverterAgent.

Purpose:
Execute deterministic model conversion workflows using explicit tool execution.

Rules:
- Never reason about CLI flags directly.
- Never assume tool arguments.
- Always discover tools using list_tools.
- Always fetch required arguments using get_tool_spec(view="required").
- Optional arguments may only be applied via explicit intent or hardcoded policy.
- All filesystem writes must occur inside the experiment workspace.
- ModelHub is read-only.
- MCP tools are stateless executors; all planning is your responsibility.
- If multiple tools match, fail with ambiguity. If none match, fail fast.

Output:
Structured JSON only (no explanations), unless explicitly asked by orchestrator.
"""
