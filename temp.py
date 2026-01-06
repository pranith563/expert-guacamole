from .config import ConverterConfig, ConverterToolNames
from .graph import build_converter_subgraph

agentic_suite/agents/converter/config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConverterToolNames:
    # ModelHub tools (namespaced keys in tool catalog)
    modelhub_get_model: str = "modelhub:modelhub_get_model"
    modelhub_get_by_path: str = "modelhub:modelhub_get_by_path"

    # Experiments tools
    experiments_create_run: str = "experiments:experiments_create_run"
    experiments_update_run_status: str = "experiments:experiments_update_run_status"
    experiments_attach_artifacts: str = "experiments:experiments_attach_artifacts"

    # QAIRT reflective registry + executor
    qairt_list_tools: str = "qairt:qairt_list_tools"
    qairt_get_tool_spec: str = "qairt:qairt_get_tool_spec"
    qairt_run: str = "qairt:qairt_run"


@dataclass(frozen=True)
class ConverterConfig:
    tools: ConverterToolNames = ConverterToolNames()

    experiments_root: str = "/data/experiments"

    # Planner loop behavior
    planner_max_tool_iters: int = 8

    # Replan loop behavior (compiler/executor -> planner)
    max_replans: int = 2

    # Tool binding mode from tool_policy.yaml (you use "convert")
    tool_mode: str = "convert"

    # Optional: prompt override path (if you want to load md at runtime)
    planner_prompt_path: Optional[str] = None

agentic_suite/agents/converter/state.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class ConversionRequest(TypedDict, total=False):
    model_ref: Dict[str, str]  # {"type":"modelhub_id"|"path","value":"..."}
    target_runtime: str
    target_soc: str
    precision: str
    quantization: str
    project_id: str
    config: Dict[str, Any]


class ConverterError(TypedDict, total=False):
    code: str
    message: str
    where: Literal["planner", "compiler", "executor"]
    step_id: Optional[str]


class ConverterState(TypedDict, total=False):
    # identity/context
    session_id: str
    user_id: str

    # in
    conversion_request: ConversionRequest
    recent_chat_summary: str

    # run/workspace
    run_id: str
    workspace_root: str
    workspace_input_model: str  # relpath under workspace

    # qairt registry snapshot
    qairt_tool_catalog: List[Dict[str, Any]]

    # planner output
    plan: Dict[str, Any]  # validated into pydantic in compiler

    # compiled plan for executor
    compiled_steps: List[Dict[str, Any]]

    # loopback controls
    replan_count: int
    max_replans: int
    replan_needed: bool
    planner_feedback: str
    last_error: ConverterError

    # final
    conversion_result: Dict[str, Any]
    last_experiment_id: str


agentic_suite/agents/converter/schemas.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: str
    tool_name: str

    required_args: Dict[str, Any] = Field(default_factory=dict)
    optional_args: Dict[str, Any] = Field(default_factory=dict)

    # workspace-relative paths for validations
    expected_outputs: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)


class ConversionPlan(BaseModel):
    status: Literal["OK", "NEEDS_INPUT"]

    questions: List[str] = Field(default_factory=list)  # when NEEDS_INPUT
    intent: Dict[str, Any] = Field(default_factory=dict)
    steps: List[PlanStep] = Field(default_factory=list)
    rationale: Optional[str] = None


class CompiledStep(BaseModel):
    step_id: str
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)

    expected_outputs: List[str] = Field(default_factory=list)  # absolute paths
    depends_on: List[str] = Field(default_factory=list)


class ConversionResult(BaseModel):
    status: Literal["COMPLETED", "FAILED", "NEEDS_INPUT"]
    run_id: str
    workspace_root: Optional[str] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    timestamp: str

agentic_suite/agents/converter/deps.py

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage

from agentic_logging import get_logger
from tool_manager import ToolManager, get_toolset_signature

logger = get_logger("converter.deps")


def now_utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def normalize_tool_key(name: str) -> str:
    # tolerate "namespace: tool" vs "namespace:tool"
    return re.sub(r"\s*:\s*", ":", name.strip())


def unwrap_tool(obj: Any) -> Any:
    # ToolManager.get_tool may return EnrichedTool or tool
    if obj is None:
        return None
    return getattr(obj, "tool", obj)


def get_tool_handle(tool_manager: ToolManager, namespaced_key: str) -> Any:
    key = normalize_tool_key(namespaced_key)
    t = tool_manager.get_tool(key)
    if t is None:
        # try spacing variant
        t = tool_manager.get_tool(key.replace(":", ": "))
    tool = unwrap_tool(t)
    if tool is None:
        raise RuntimeError(f"Tool not found in catalog: {namespaced_key}")
    return tool


def safe_join(workspace_root: str, rel: str) -> str:
    if os.path.isabs(rel):
        raise ValueError(f"Absolute paths are not allowed in plan args: {rel}")
    norm = os.path.normpath(rel)
    if norm.startswith("..") or "/../" in norm.replace("\\", "/"):
        raise ValueError(f"Path traversal not allowed: {rel}")

    abs_path = os.path.abspath(os.path.join(workspace_root, norm))
    root_abs = os.path.abspath(workspace_root)
    if not abs_path.startswith(root_abs + os.sep) and abs_path != root_abs:
        raise ValueError(f"Path escapes workspace root: {rel}")
    return abs_path


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Expected JSON object in planner output")
        return json.loads(text[start:end + 1])


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def discover_artifacts(workspace_root: str) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    converted_dir = os.path.join(workspace_root, "converted")
    if not os.path.isdir(converted_dir):
        return artifacts

    for root, _, files in os.walk(converted_dir):
        for fn in files:
            abs_path = os.path.join(root, fn)
            rel = os.path.relpath(abs_path, workspace_root)
            typ = "file"
            if fn.endswith(".dlc"):
                typ = "dlc"
            artifacts.append({"type": typ, "path": rel})
    return artifacts


class ToolBindingCache:
    """
    Mirrors your ChatAgent approach:
      llm.bind_tools(tools) cached by (mode, signature).
    """

    def __init__(self, llm: Any, tool_manager: ToolManager):
        self.llm = llm
        self.tool_manager = tool_manager
        self._cache: Dict[Tuple[str, int], Any] = {}

    def get_llm_for_mode(self, mode: str) -> Any:
        tools = self.tool_manager.get_tools_for_mode(mode)
        signature = get_toolset_signature(tools)
        key = (mode, signature)
        if key not in self._cache:
            logger.info(f"Binding tools for mode='{mode}' signature={signature}")
            self._cache[key] = self.llm.bind_tools(tools)
        return self._cache[key]


agentic_suite/agents/converter/prompts/converter_planner.md

You are ConverterPlanner.

You are planning a model conversion using QAIRT tools (SNPE/QNN).
You may call tools to fetch tool lists and tool specs.

CRITICAL RULES:
- NEVER call any execution tool that runs conversion (e.g., qairt_run). Planning only.
- Use tools only to understand available conversion tools and their required args.
- Your output MUST be a single JSON object (no markdown) matching the required schema below.

INPUT:
You will receive a JSON context containing:
- conversion_request (user intent)
- workspace_root (absolute)
- workspace_input_model (workspace-relative path to input model)
- qairt_tool_catalog (list of available QAIRT tools; may be empty if not fetched yet)
- last_error / planner_feedback if this is a replan attempt

YOUR OUTPUT SCHEMA:
{
  "status": "OK" | "NEEDS_INPUT",
  "questions": [ ... ],                 // if NEEDS_INPUT
  "intent": { ... },                    // summarize intent (target_runtime, target_soc, precision...)
  "steps": [
    {
      "step_id": "convert",
      "tool_name": "<UNDERLYING_QAIRT_TOOL_NAME>",
      "required_args": { "<arg>": "<value>", ... },
      "optional_args": { "<arg>": "<value>", ... },
      "expected_outputs": [ "converted/model.dlc", ... ],
      "depends_on": []
    }
  ],
  "rationale": "short reason"
}

IMPORTANT PATH RULES:
- Any file/dir args MUST be workspace-relative (examples: "input/model.onnx", "converted/model.dlc", "converted/cached/htp").
- Do NOT use absolute paths in required_args/optional_args.

REQUIRED ARGS:
- You MUST fetch qairt_get_tool_spec(view="required") for the chosen tool_name.
- Populate ALL required args exactly (names must match spec).
- Use workspace_input_model for the model input path whenever applicable.

REPLAN:
- If planner_feedback says the compiler/executor failed, adjust the plan accordingly.

agentic_suite/agents/converter/planner.py


from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode

from agentic_logging import get_logger
from .config import ConverterConfig
from .state import ConverterState
from .deps import (
    now_utc_iso,
    extract_json_object,
    get_tool_handle,
    write_json,
    ToolBindingCache,
)

logger = get_logger("converter.planner")


def load_planner_prompt(config: ConverterConfig) -> str:
    if config.planner_prompt_path and os.path.exists(config.planner_prompt_path):
        with open(config.planner_prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    # fallback to packaged prompt file path relative to this file
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "converter_planner.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


class PlannerNode:
    """
    Planner node does two things:
      1) Deterministic init (run creation + workspace + input model copy) if not done.
      2) ReAct planning using ToolNode (LLM tool calling) -> produces plan JSON in state["plan"].

    Tool execution happens via ToolNode created from ToolManager tools for mode=config.tool_mode.
    """

    def __init__(
        self,
        *,
        llm_cache: ToolBindingCache,
        tool_manager: Any,
        config: ConverterConfig,
    ):
        self.llm_cache = llm_cache
        self.tool_manager = tool_manager
        self.config = config
        self.prompt = load_planner_prompt(config)

    async def __call__(self, state: ConverterState) -> ConverterState:
        session_id = state.get("session_id", "unknown")
        state.setdefault("replan_count", 0)
        state.setdefault("max_replans", self.config.max_replans)
        state["replan_needed"] = False

        # deterministic init
        if not state.get("run_id"):
            await self._init_run_and_workspace(state)

        # preload qairt tool catalog (helpful for planner; not required)
        if not state.get("qairt_tool_catalog"):
            try:
                qairt_list = get_tool_handle(self.tool_manager, self.config.tools.qairt_list_tools)
                catalog = await qairt_list.ainvoke({})
                if isinstance(catalog, dict) and "tools" in catalog:
                    catalog = catalog["tools"]
                state["qairt_tool_catalog"] = catalog if isinstance(catalog, list) else []
            except Exception as e:
                logger.warning(f"[{session_id}] Could not preload qairt tool catalog: {e}")
                state["qairt_tool_catalog"] = []

        # If a previous node asked for replan, planner_feedback is set
        ctx = {
            "session_id": session_id,
            "user_id": state.get("user_id"),
            "conversion_request": state.get("conversion_request", {}),
            "workspace_root": state.get("workspace_root"),
            "workspace_input_model": state.get("workspace_input_model"),
            "qairt_tool_catalog": state.get("qairt_tool_catalog", []),
            "replan_count": state.get("replan_count", 0),
            "last_error": state.get("last_error"),
            "planner_feedback": state.get("planner_feedback"),
        }

        llm = self.llm_cache.get_llm_for_mode(self.config.tool_mode)
        tool_node = ToolNode(self.tool_manager.get_tools_for_mode(self.config.tool_mode))

        messages: List[BaseMessage] = [
            SystemMessage(content=self.prompt),
            HumanMessage(content=json.dumps(ctx, indent=2)),
        ]

        for _ in range(self.config.planner_max_tool_iters):
            ai: AIMessage = await llm.ainvoke(messages)
            messages.append(ai)

            tool_calls = getattr(ai, "tool_calls", None) or []
            if tool_calls:
                tool_out = await tool_node.ainvoke({"messages": messages})
                tool_messages = tool_out.get("messages", [])
                if tool_messages:
                    messages.extend(tool_messages)
                continue

            # final plan JSON
            payload = extract_json_object(ai.content or "")
            state["plan"] = payload
            return state

        # max iters
        state["last_error"] = {"code": "PLANNER_MAX_ITERS", "message": "Planner exceeded max tool iterations", "where": "planner"}
        state["conversion_result"] = {
            "status": "FAILED",
            "run_id": state.get("run_id", ""),
            "error": state["last_error"],
            "timestamp": now_utc_iso(),
        }
        state["last_experiment_id"] = state.get("run_id", "")
        return state

    async def _init_run_and_workspace(self, state: ConverterState) -> None:
        """
        Deterministic init:
          - resolve model from ModelHub
          - create run in experiments
          - create workspace dirs
          - copy model into workspace/input
        """
        req = state.get("conversion_request") or {}
        model_ref = req.get("model_ref") or {}
        if "type" not in model_ref or "value" not in model_ref:
            raise RuntimeError("conversion_request.model_ref must be {type,value}")

        user_id = state.get("user_id") or "default_user"

        mh_get_model = get_tool_handle(self.tool_manager, self.config.tools.modelhub_get_model)
        mh_get_by_path = get_tool_handle(self.tool_manager, self.config.tools.modelhub_get_by_path)

        if model_ref["type"] == "modelhub_id":
            model = await mh_get_model.ainvoke({"model_id": model_ref["value"]})
        else:
            model = await mh_get_by_path.ainvoke({"file_path": model_ref["value"]})

        model_id = model.get("id") or model.get("model_id") or model.get("name") or "unknown_model"
        filename = model.get("filename") or os.path.basename(model.get("absolute_path", "")) or "model.bin"
        abs_path = model.get("absolute_path") or model.get("path")
        if not abs_path:
            raise RuntimeError(f"ModelHub response missing absolute_path/path. Keys={list(model.keys())}")

        exp_create = get_tool_handle(self.tool_manager, self.config.tools.experiments_create_run)
        exp_update = get_tool_handle(self.tool_manager, self.config.tools.experiments_update_run_status)

        run_cfg = {
            "conversion_request": req,
            "model_ref": model_ref,
            "model_id": model_id,
            "created_at": now_utc_iso(),
        }

        resp = await exp_create.ainvoke({
            "run_type": "convert",
            "user_id": user_id,
            "config": run_cfg,
            "project_id": req.get("project_id"),
            "model_id": model_id,
        })

        run_id = resp.get("run_id") or resp.get("id") or str(uuid.uuid4())
        state["run_id"] = run_id
        state["last_experiment_id"] = run_id

        workspace_root = os.path.join(self.config.experiments_root, user_id, run_id)
        state["workspace_root"] = workspace_root

        os.makedirs(os.path.join(workspace_root, "input"), exist_ok=True)
        os.makedirs(os.path.join(workspace_root, "converted"), exist_ok=True)
        os.makedirs(os.path.join(workspace_root, "logs"), exist_ok=True)

        dst_abs = os.path.join(workspace_root, "input", filename)
        shutil.copy2(abs_path, dst_abs)
        state["workspace_input_model"] = os.path.relpath(dst_abs, workspace_root)

        try:
            await exp_update.ainvoke({"run_id": run_id, "status": "WORKSPACE_READY", "error_message": None})
        except Exception:
            pass

        write_json(os.path.join(workspace_root, "metadata.json"), {
            "run_id": run_id,
            "user_id": user_id,
            "model_id": model_id,
            "source_abs_path": abs_path,
            "workspace_input_model": state["workspace_input_model"],
            "created_at": now_utc_iso(),
        })

agentic_suite/agents/converter/compiler.py

from __future__ import annotations

import os
from typing import Any, Dict, List

from agentic_logging import get_logger
from .config import ConverterConfig
from .state import ConverterState
from .schemas import ConversionPlan, CompiledStep
from .deps import (
    now_utc_iso,
    get_tool_handle,
    safe_join,
    write_json,
)

logger = get_logger("converter.compiler")


class CompilerNode:
    """
    Deterministic compiler:
      - Validates planner plan against tool spec
      - Produces compiled_steps with absolute paths for args and outputs
      - On failure sets replan_needed + planner_feedback (bounded in graph router)
    """

    def __init__(self, *, tool_manager: Any, config: ConverterConfig):
        self.tool_manager = tool_manager
        self.config = config

    async def __call__(self, state: ConverterState) -> ConverterState:
        state["replan_needed"] = False

        # If planner already produced a terminal result, skip
        if state.get("conversion_result", {}).get("status") in ("FAILED", "NEEDS_INPUT", "COMPLETED"):
            return state

        workspace_root = state.get("workspace_root")
        run_id = state.get("run_id")
        if not workspace_root or not run_id:
            return self._fail(state, "SPEC_MISSING", "workspace_root/run_id missing")

        plan_raw = state.get("plan")
        if not plan_raw:
            return await self._loopback(state, "SPEC_MISSING", "plan missing from state")

        # validate plan via pydantic
        try:
            plan = ConversionPlan.model_validate(plan_raw)
        except Exception as e:
            return await self._loopback(state, "PLAN_INVALID", f"plan schema invalid: {e}")

        if plan.status == "NEEDS_INPUT":
            state["conversion_result"] = {
                "status": "NEEDS_INPUT",
                "run_id": run_id,
                "questions": plan.questions,
                "timestamp": now_utc_iso(),
            }
            state["last_experiment_id"] = run_id
            return state

        if not plan.steps:
            return await self._loopback(state, "SPEC_MISSING", "plan.steps empty")

        qairt_get_spec = get_tool_handle(self.tool_manager, self.config.tools.qairt_get_tool_spec)

        compiled: List[Dict[str, Any]] = []
        try:
            for step in plan.steps:
                req_spec = await qairt_get_spec.ainvoke({"tool_name": step.tool_name, "view": "required"})
                opt_spec = await qairt_get_spec.ainvoke({"tool_name": step.tool_name, "view": "optional"})

                req_names = [a.get("name") for a in (req_spec.get("args") or []) if a.get("name")]
                missing = [n for n in req_names if n not in step.required_args]
                if missing:
                    raise ValueError(f"Step '{step.step_id}' missing required args: {missing}")

                opt_names = {a.get("name") for a in (opt_spec.get("args") or []) if a.get("name")}
                unknown_opt = [k for k in step.optional_args.keys() if k not in opt_names]
                if unknown_opt:
                    raise ValueError(f"Step '{step.step_id}' unsupported optional args: {unknown_opt}")

                def compile_value(v: Any) -> Any:
                    if isinstance(v, str):
                        # treat as workspace-relative path by heuristic
                        if (
                            v.startswith(("input/", "converted/", "logs/"))
                            or "/" in v or "\\" in v
                            or v.endswith((".onnx", ".pb", ".tflite", ".dlc"))
                        ):
                            return safe_join(workspace_root, v)
                    return v

                args: Dict[str, Any] = {}
                for k, v in step.required_args.items():
                    args[k] = compile_value(v)
                for k, v in step.optional_args.items():
                    args[k] = compile_value(v)

                expected_abs = [safe_join(workspace_root, p) for p in step.expected_outputs]

                compiled_step = CompiledStep(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    args=args,
                    expected_outputs=expected_abs,
                    depends_on=step.depends_on,
                )
                compiled.append(compiled_step.model_dump())

            state["compiled_steps"] = compiled

            # persist plan and compiled info
            write_json(os.path.join(workspace_root, "plan.json"), plan_raw)
            write_json(os.path.join(workspace_root, "compiled_steps.json"), {"steps": compiled})
            return state

        except Exception as e:
            return await self._loopback(state, "COMPILER_ERROR", str(e))

    async def _loopback(self, state: ConverterState, code: str, message: str) -> ConverterState:
        state["last_error"] = {"code": code, "message": message, "where": "compiler"}
        state["replan_needed"] = True
        state["planner_feedback"] = (
            f"Compiler failed. code={code}. message={message}. "
            "Fix the plan JSON: choose correct tool_name and ensure required_args match tool spec; "
            "ensure workspace-relative paths."
        )
        return state

    def _fail(self, state: ConverterState, code: str, message: str) -> ConverterState:
        run_id = state.get("run_id", "")
        state["conversion_result"] = {
            "status": "FAILED",
            "run_id": run_id,
            "error": {"code": code, "message": message, "where": "compiler"},
            "timestamp": now_utc_iso(),
        }
        state["last_experiment_id"] = run_id
        return state

agentic_suite/agents/converter/executor.py

from __future__ import annotations

import os
from typing import Any, Dict, List

from agentic_logging import get_logger
from .config import ConverterConfig
from .state import ConverterState
from .deps import (
    now_utc_iso,
    discover_artifacts,
    get_tool_handle,
    write_json,
)

logger = get_logger("converter.executor")


class ExecutorNode:
    """
    Deterministic executor:
      - Executes compiled steps sequentially using qairt_run
      - Validates expected outputs
      - Updates experiments status + attaches artifacts
      - On execution failure sets replan_needed + planner_feedback (bounded in graph router)
    """

    def __init__(self, *, tool_manager: Any, config: ConverterConfig):
        self.tool_manager = tool_manager
        self.config = config

    async def __call__(self, state: ConverterState) -> ConverterState:
        state["replan_needed"] = False

        # If terminal already, skip
        if state.get("conversion_result", {}).get("status") in ("FAILED", "NEEDS_INPUT", "COMPLETED"):
            return state

        workspace_root = state.get("workspace_root")
        run_id = state.get("run_id")
        if not workspace_root or not run_id:
            return self._fail(state, "SPEC_MISSING", "workspace_root/run_id missing")

        compiled_steps = state.get("compiled_steps") or []
        if not compiled_steps:
            return await self._loopback(state, "SPEC_MISSING", "compiled_steps missing")

        qairt_run = get_tool_handle(self.tool_manager, self.config.tools.qairt_run)
        exp_update = get_tool_handle(self.tool_manager, self.config.tools.experiments_update_run_status)
        exp_attach = get_tool_handle(self.tool_manager, self.config.tools.experiments_attach_artifacts)

        logs_dir = os.path.join(workspace_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # best-effort set RUNNING
        try:
            await exp_update.ainvoke({"run_id": run_id, "status": "RUNNING", "error_message": None})
        except Exception:
            pass

        try:
            for step in compiled_steps:
                step_id = step["step_id"]
                tool_name = step["tool_name"]
                args = step["args"]
                expected_outputs = step.get("expected_outputs") or []

                try:
                    await exp_update.ainvoke({"run_id": run_id, "status": f"STEP:{step_id}", "error_message": None})
                except Exception:
                    pass

                exec_result = await qairt_run.ainvoke({
                    "tool_name": tool_name,
                    "args": args,
                    "workspace_root": workspace_root,
                })

                write_json(os.path.join(logs_dir, f"{step_id}.exec_result.json"), exec_result)

                exit_code = exec_result.get("exit_code", 0)
                if exit_code not in (0, "0", None):
                    raise RuntimeError(f"Step '{step_id}' failed (exit_code={exit_code})")

                for outp in expected_outputs:
                    if not os.path.exists(outp):
                        raise RuntimeError(f"Expected output missing after step '{step_id}': {outp}")

            artifacts = discover_artifacts(workspace_root)

            try:
                await exp_attach.ainvoke({"run_id": run_id, "artifacts": artifacts})
            except Exception:
                pass

            try:
                await exp_update.ainvoke({"run_id": run_id, "status": "COMPLETED", "error_message": None})
            except Exception:
                pass

            state["conversion_result"] = {
                "status": "COMPLETED",
                "run_id": run_id,
                "workspace_root": workspace_root,
                "artifacts": artifacts,
                "timestamp": now_utc_iso(),
            }
            state["last_experiment_id"] = run_id
            return state

        except Exception as e:
            return await self._loopback(state, "EXECUTOR_ERROR", str(e))

    async def _loopback(self, state: ConverterState, code: str, message: str) -> ConverterState:
        state["last_error"] = {"code": code, "message": message, "where": "executor"}
        state["replan_needed"] = True
        state["planner_feedback"] = (
            f"Executor failed. code={code}. message={message}. "
            "Fix the plan JSON (tool_name/args/expected_outputs). "
            "Ensure expected_outputs are correct and paths are workspace-relative."
        )
        return state

    def _fail(self, state: ConverterState, code: str, message: str) -> ConverterState:
        run_id = state.get("run_id", "")
        state["conversion_result"] = {
            "status": "FAILED",
            "run_id": run_id,
            "error": {"code": code, "message": message, "where": "executor"},
            "timestamp": now_utc_iso(),
        }
        state["last_experiment_id"] = run_id
        return state

agentic_suite/agents/converter/graph.py

from __future__ import annotations

from typing import Literal, Any

from langgraph.graph import StateGraph, START, END

from agentic_logging import get_logger
from .state import ConverterState
from .config import ConverterConfig
from .deps import ToolBindingCache
from .planner import PlannerNode
from .compiler import CompilerNode
from .executor import ExecutorNode

logger = get_logger("converter.graph")


Route = Literal["planner", "compiler", "executor", "__end__"]


def _route_after_compiler(state: ConverterState) -> Route:
    # If compiler asked for replan and we have replans left -> planner
    if state.get("replan_needed"):
        if int(state.get("replan_count", 0)) < int(state.get("max_replans", 0)):
            state["replan_count"] = int(state.get("replan_count", 0)) + 1
            return "planner"
        return "__end__"
    return "executor"


def _route_after_executor(state: ConverterState) -> Route:
    if state.get("replan_needed"):
        if int(state.get("replan_count", 0)) < int(state.get("max_replans", 0)):
            state["replan_count"] = int(state.get("replan_count", 0)) + 1
            return "planner"
        return "__end__"
    return "__end__"


def build_converter_subgraph(*, llm: Any, tool_manager: Any, config: ConverterConfig):
    """
    Builds and compiles the converter subgraph:
      planner -> compiler -> executor
      with loopback edges to planner on compiler/executor errors (bounded).
    """
    llm_cache = ToolBindingCache(llm=llm, tool_manager=tool_manager)

    planner = PlannerNode(llm_cache=llm_cache, tool_manager=tool_manager, config=config)
    compiler = CompilerNode(tool_manager=tool_manager, config=config)
    executor = ExecutorNode(tool_manager=tool_manager, config=config)

    g = StateGraph(ConverterState)

    g.add_node("planner", planner)
    g.add_node("compiler", compiler)
    g.add_node("executor", executor)

    g.add_edge(START, "planner")
    g.add_edge("planner", "compiler")

    g.add_conditional_edges(
        "compiler",
        _route_after_compiler,
        {
            "planner": "planner",
            "executor": "executor",
            "__end__": END,
        },
    )

    g.add_conditional_edges(
        "executor",
        _route_after_executor,
        {
            "planner": "planner",
            "__end__": END,
        },
    )

    return g.compile()

agentic_suite/agents/converter_agent.py

from __future__ import annotations

from typing import Any

from agentic_logging import get_logger
from agentic_suite.graph.state import GraphState  # your main graph state type
from .converter.config import ConverterConfig
from .converter.graph import build_converter_subgraph
from .converter.state import ConverterState

logger = get_logger("converter_agent")


class ConverterAgent:
    """
    Main graph node wrapper around converter subgraph.

    Supervisor should set:
      state["active_task_type"] = "convert"
      state["conversion_request"] = {...}  # preferred
    """

    def __init__(self, *, llm: Any, tool_manager: Any, config: ConverterConfig | None = None):
        self.llm = llm
        self.tool_manager = tool_manager
        self.config = config or ConverterConfig()
        self.subgraph = build_converter_subgraph(llm=self.llm, tool_manager=self.tool_manager, config=self.config)

    async def __call__(self, state: GraphState) -> GraphState:
        session_id = state.get("session_id") or state.get("thread_id") or "unknown"
        user_id = state.get("user_id") or state.get("user") or "default_user"

        conversion_request = state.get("conversion_request") or state.get("conversion_intent")
        if not conversion_request:
            state["conversion_result"] = {
                "status": "FAILED",
                "error": {"code": "MISSING_CONVERSION_REQUEST", "message": "No conversion_request in GraphState"},
            }
            return state

        conv_state: ConverterState = {
            "session_id": session_id,
            "user_id": user_id,
            "conversion_request": conversion_request,
            "recent_chat_summary": state.get("recent_chat_summary", ""),
            "replan_count": 0,
            "max_replans": self.config.max_replans,
            "replan_needed": False,
        }

        out: ConverterState = await self.subgraph.ainvoke(conv_state)

        state["conversion_result"] = out.get("conversion_result", {})
        state["last_experiment_id"] = out.get("run_id")
        return state





