from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage


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

    # ReAct messages inside subgraph (needed for ToolNode routing)
    messages: List[BaseMessage]
    messages_initialized: bool

    # planner output + flags
    plan: Dict[str, Any]
    planner_done: bool

    # compiled plan for executor
    compiled_steps: List[Dict[str, Any]]

    # loopback controls
    replan_count: int
    max_replans: int
    replan_needed: bool
    planner_feedback: str
    planner_feedback_consumed: bool
    last_error: ConverterError

    # final
    conversion_result: Dict[str, Any]
    last_experiment_id: str


from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

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

    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "converter_planner.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


class PlannerNode:
    """
    Planner node (ReAct style, one step per graph tick):
      - ensures deterministic init (run + workspace + input copy) once
      - ensures state["messages"] exists
      - appends planner_feedback when replanning
      - calls LLM once and appends AIMessage to messages
      - if AIMessage contains tool_calls -> routing will go to convert_tools ToolNode
      - else parses final plan JSON and sets planner_done=True
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
        state.setdefault("replan_needed", False)
        state.setdefault("planner_done", False)

        # deterministic init
        if not state.get("run_id"):
            await self._init_run_and_workspace(state)

        # messages init (required for ToolNode routing)
        if not state.get("messages_initialized"):
            state["messages"] = []
            ctx = {
                "session_id": session_id,
                "user_id": state.get("user_id"),
                "conversion_request": state.get("conversion_request", {}),
                "workspace_root": state.get("workspace_root"),
                "workspace_input_model": state.get("workspace_input_model"),
                "recent_chat_summary": state.get("recent_chat_summary", ""),
            }
            state["messages"].append(SystemMessage(content=self.prompt))
            state["messages"].append(HumanMessage(content=json.dumps(ctx, indent=2)))
            state["messages_initialized"] = True

        # If compiler/executor requested replan, inject feedback exactly once
        if state.get("planner_feedback") and not state.get("planner_feedback_consumed"):
            fb = {
                "planner_feedback": state.get("planner_feedback"),
                "last_error": state.get("last_error"),
                "instruction": (
                    "Replan now. Output ONLY valid plan JSON. "
                    "You may call tools (qairt_list_tools / qairt_get_tool_spec) if needed."
                ),
            }
            state["messages"].append(HumanMessage(content=json.dumps(fb, indent=2)))
            state["planner_feedback_consumed"] = True
            state["planner_done"] = False
            state.pop("plan", None)

        # Call LLM once (tool calls will be handled by convert_tools node via edges)
        llm = self.llm_cache.get_llm_for_mode(self.config.tool_mode)
        ai: AIMessage = await llm.ainvoke(state["messages"])
        state["messages"].append(ai)

        # If tool_calls exist, stop here; graph routing will send to ToolNode
        tool_calls = getattr(ai, "tool_calls", None) or []
        if tool_calls:
            return state

        # Otherwise expect final plan JSON
        try:
            payload = extract_json_object(ai.content or "")
            state["plan"] = payload
            # planner done means route to compiler
            state["planner_done"] = True
            return state
        except Exception as e:
            state["last_error"] = {"code": "PLAN_PARSE_ERROR", "message": str(e), "where": "planner"}
            state["replan_needed"] = True
            state["planner_feedback"] = (
                "Your last output was not valid JSON. Output ONLY a valid plan JSON object "
                "matching the required schema."
            )
            state["planner_feedback_consumed"] = False
            return state

    async def _init_run_and_workspace(self, state: ConverterState) -> None:
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
    def __init__(self, *, tool_manager: Any, config: ConverterConfig):
        self.tool_manager = tool_manager
        self.config = config

    async def __call__(self, state: ConverterState) -> ConverterState:
        state["replan_needed"] = False

        # Terminal cases
        if state.get("conversion_result", {}).get("status") in ("FAILED", "NEEDS_INPUT", "COMPLETED"):
            return state

        workspace_root = state.get("workspace_root")
        run_id = state.get("run_id")
        if not workspace_root or not run_id:
            return self._fail(state, "SPEC_MISSING", "workspace_root/run_id missing")

        plan_raw = state.get("plan")
        if not plan_raw:
            return await self._loopback(state, "SPEC_MISSING", "plan missing from state")

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
            "Fix the plan JSON: ensure tool_name is correct, required_args match tool spec, "
            "and all paths are workspace-relative."
        )
        state["planner_feedback_consumed"] = False
        state["planner_done"] = False
        state.pop("compiled_steps", None)
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
    def __init__(self, *, tool_manager: Any, config: ConverterConfig):
        self.tool_manager = tool_manager
        self.config = config

    async def __call__(self, state: ConverterState) -> ConverterState:
        state["replan_needed"] = False

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
            "Fix the plan JSON (tool_name/args/expected_outputs). Ensure paths are workspace-relative."
        )
        state["planner_feedback_consumed"] = False
        state["planner_done"] = False
        state.pop("compiled_steps", None)
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

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agentic_logging import get_logger
from .state import ConverterState
from .config import ConverterConfig
from .deps import ToolBindingCache
from .planner import PlannerNode
from .compiler import CompilerNode
from .executor import ExecutorNode

logger = get_logger("converter.graph")

RouteAfterPlanner = Literal["convert_tools", "compiler", "__end__"]
RouteAfterCompiler = Literal["planner", "executor", "__end__"]
RouteAfterExecutor = Literal["planner", "__end__"]


def _route_after_planner(state: ConverterState) -> RouteAfterPlanner:
    # Terminal
    status = (state.get("conversion_result") or {}).get("status")
    if status in ("FAILED", "NEEDS_INPUT", "COMPLETED"):
        return "__end__"

    # If tool calls are present, go to ToolNode
    if tools_condition(state) == "tools":
        return "convert_tools"

    # If planner produced a plan, go compiler
    if state.get("planner_done"):
        return "compiler"

    # Otherwise end (or you can loop back to planner)
    return "__end__"


def _route_after_compiler(state: ConverterState) -> RouteAfterCompiler:
    status = (state.get("conversion_result") or {}).get("status")
    if status in ("FAILED", "NEEDS_INPUT", "COMPLETED"):
        return "__end__"

    if state.get("replan_needed"):
        if int(state.get("replan_count", 0)) < int(state.get("max_replans", 0)):
            state["replan_count"] = int(state.get("replan_count", 0)) + 1
            return "planner"
        return "__end__"

    return "executor"


def _route_after_executor(state: ConverterState) -> RouteAfterExecutor:
    status = (state.get("conversion_result") or {}).get("status")
    if status in ("FAILED", "NEEDS_INPUT", "COMPLETED"):
        return "__end__"

    if state.get("replan_needed"):
        if int(state.get("replan_count", 0)) < int(state.get("max_replans", 0)):
            state["replan_count"] = int(state.get("replan_count", 0)) + 1
            return "planner"
        return "__end__"

    return "__end__"


def build_converter_subgraph(*, llm: Any, tool_manager: Any, config: ConverterConfig):
    """
    Subgraph:
      planner -> (convert_tools -> planner)* -> compiler -> (planner)* -> executor -> (planner)* -> END
    """
    llm_cache = ToolBindingCache(llm=llm, tool_manager=tool_manager)

    planner = PlannerNode(llm_cache=llm_cache, tool_manager=tool_manager, config=config)
    compiler = CompilerNode(tool_manager=tool_manager, config=config)
    executor = ExecutorNode(tool_manager=tool_manager, config=config)

    # Converter tool node (mode-scoped tools)
    convert_tools = ToolNode(tool_manager.get_tools_for_mode(config.tool_mode))

    g = StateGraph(ConverterState)

    g.add_node("planner", planner)
    g.add_node("convert_tools", convert_tools)
    g.add_node("compiler", compiler)
    g.add_node("executor", executor)

    g.add_edge(START, "planner")

    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "convert_tools": "convert_tools",
            "compiler": "compiler",
            "__end__": END,
        },
    )

    # tools always go back to planner
    g.add_edge("convert_tools", "planner")

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
    

