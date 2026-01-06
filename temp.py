from .schemas import ConversionContext, ConversionPlanIR, CompiledPlan
from .planner import ConverterPlanner
from .compiler import PlanCompiler
from .executor import PlanExecutor

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ConverterToolNames:
    # ModelHub (read-only)
    modelhub_get_model: str
    modelhub_get_by_path: str

    # SNPE/QNN reflective registry
    list_tools: str
    get_tool_spec: str
    run_tool: str


@dataclass(frozen=True)
class ConverterPaths:
    experiments_root: str = "/data/experiments"


@dataclass(frozen=True)
class ConverterPolicyConfig:
    """
    Minimal policy buckets. Expand over time.
    The compiler only applies args that are present in tool specs.
    """
    policy_version: str = "2026-01-06"
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = None  # runtime -> op -> bucket -> optional_args

    def __post_init__(self):
        if self.buckets is None:
            object.__setattr__(self, "buckets", {
                "snpe": {
                    "convert_model": {
                        "safe": {"precision": "fp16"},
                        "balanced": {"precision": "fp16"},
                        "fast": {"precision": "fp16"},
                        "max_perf": {"precision": "fp16"},
                    },
                    "cache_model": {
                        "safe": {},
                        "balanced": {"accelerator": "dsp"},
                        "fast": {"accelerator": "htp"},
                        "max_perf": {"accelerator": "htp"},
                    },
                    "validate_model": {
                        "safe": {},
                        "balanced": {},
                        "fast": {},
                        "max_perf": {},
                    },
                },
                "qnn": {
                    "convert_model": {
                        "safe": {"precision": "fp16"},
                        "balanced": {"precision": "fp16"},
                        "fast": {"precision": "fp16"},
                        "max_perf": {"precision": "fp16"},
                    },
                    "cache_model": {
                        "safe": {},
                        "balanced": {},
                        "fast": {},
                        "max_perf": {},
                    },
                    "validate_model": {
                        "safe": {},
                        "balanced": {},
                        "fast": {},
                        "max_perf": {},
                    },
                },
            })


@dataclass(frozen=True)
class ConverterConfig:
    tool_names: ConverterToolNames
    paths: ConverterPaths = ConverterPaths()
    policy: ConverterPolicyConfig = ConverterPolicyConfig()

    # Safety / planner loop limits
    planner_max_iters: int = 8
    max_tools_in_catalog: int = 2000


--------////

from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# -----------------------
# Planner input
# -----------------------

class ToolCatalogEntry(BaseModel):
    name: str
    category: str
    source: Optional[str] = None
    target: Optional[str] = None
    accelerator: Optional[str] = None
    sdk: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class PreviousExperimentSummary(BaseModel):
    experiment_id: str
    status: str
    model_id: Optional[str] = None
    artifacts: List[Dict[str, str]] = Field(default_factory=list)
    intent: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ModelRef(BaseModel):
    type: Literal["modelhub_id", "path"]
    value: str


class ResolvedModelInfo(BaseModel):
    model_id: str
    filename: str
    absolute_path: str
    # optional extras
    framework: Optional[str] = None
    format: Optional[str] = None


class ConversionContext(BaseModel):
    session_id: str
    user: str
    latest_user_message: str
    recent_chat_summary: Optional[str] = None

    model_ref: ModelRef
    resolved_model: Optional[ResolvedModelInfo] = None

    tool_catalog: List[ToolCatalogEntry] = Field(default_factory=list)
    previous_experiments: List[PreviousExperimentSummary] = Field(default_factory=list)

    default_policy_bucket: Literal["safe", "balanced", "fast", "max_perf"] = "balanced"
    constraints: Dict[str, Any] = Field(default_factory=dict)
    hints: Dict[str, Any] = Field(default_factory=dict)


# -----------------------
# Planner output IR
# -----------------------

class IntentIR(BaseModel):
    model_ref: ModelRef
    source_framework: Literal["tensorflow", "pytorch", "onnx", "tflite"]
    target_runtime: Literal["snpe", "qnn", "tflite", "mdla"]
    target_soc: str
    precision: Literal["fp32", "fp16", "int8"] = "fp16"
    quantization: Literal["none", "static", "dynamic"] = "none"


class ToolSelector(BaseModel):
    category: Literal["conversion", "caching", "validation", "export"]
    source: Optional[str] = None
    target: Optional[str] = None
    accelerator: Optional[str] = None
    tags_all: List[str] = Field(default_factory=list)
    tags_any: List[str] = Field(default_factory=list)
    sdk: Optional[str] = None


class StepIO(BaseModel):
    inputs: Dict[str, str] = Field(default_factory=dict)   # workspace-relative
    outputs: Dict[str, str] = Field(default_factory=dict)  # workspace-relative


class StepIR(BaseModel):
    step_id: str
    op: Literal["convert_model", "cache_model", "validate_model", "export_artifacts"]
    tool_selector: ToolSelector
    io: StepIO = Field(default_factory=StepIO)
    params: Dict[str, Any] = Field(default_factory=dict)  # intent-level knobs only
    policy_bucket: Optional[Literal["safe", "balanced", "fast", "max_perf"]] = None
    depends_on: List[str] = Field(default_factory=list)


class NeedsInput(BaseModel):
    questions: List[str]
    missing_fields: List[str] = Field(default_factory=list)


class ConversionPlanIR(BaseModel):
    plan_version: str = "1"
    status: Literal["OK", "NEEDS_INPUT"] = "OK"
    needs_input: Optional[NeedsInput] = None
    intent: IntentIR
    steps: List[StepIR] = Field(default_factory=list)
    rationale: Optional[str] = None


# -----------------------
# Compiled plan (deterministic)
# -----------------------

class CompiledStep(BaseModel):
    step_id: str
    op: str
    tool_name: str
    tool_category: str

    args: Dict[str, Any] = Field(default_factory=dict)
    abs_inputs: Dict[str, str] = Field(default_factory=dict)
    abs_outputs: Dict[str, str] = Field(default_factory=dict)

    depends_on: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)

    skip_if_outputs_exist: bool = True
    validations: List[Dict[str, Any]] = Field(default_factory=list)


class CompiledPlan(BaseModel):
    compiled_version: str = "1"
    experiment_id: str
    workspace_root: str

    intent: Dict[str, Any]
    tool_catalog_snapshot: List[Dict[str, Any]] = Field(default_factory=list)

    steps: List[CompiledStep] = Field(default_factory=list)

    policy_version: Optional[str] = None
    compiler_metadata: Dict[str, Any] = Field(default_factory=dict)


# -----------------------
# Result (for main graph)
# -----------------------

class ConversionResult(BaseModel):
    experiment_id: str
    status: Literal["COMPLETED", "FAILED", "NEEDS_INPUT"]
    artifacts: List[Dict[str, str]] = Field(default_factory=list)
    error: Optional[str] = None
    needs_input: Optional[NeedsInput] = None


-------///

CONVERTER_PLANNER_SYSTEM = """\
You are ConverterPlanner.

Goal:
Produce a ConversionPlanIR that converts a model to the requested target runtime and SoC.

You are a ReAct-style planner. You may call ONLY these tools:
- list_tools
- get_tool_spec
- modelhub_get_model / modelhub_get_by_path (optional)

Rules:
- Never call run_tool or execute conversion.
- Never output CLI flags directly; work at intent/policy level.
- Prefer semantic tool selection via tool_selector; do not hardcode tool names.
- All inputs/outputs in the plan must be workspace-relative paths (no absolute paths, no .. traversal).
- If info is missing, return NEEDS_INPUT with specific questions.
- If multiple tools match, refine selector or return NEEDS_INPUT/ambiguity.

Output:
Return ONE JSON object matching ConversionPlanIR. No extra text.
"""

-----/////


from __future__ import annotations

from typing import Any, Dict, List, Protocol


class AsyncTool(Protocol):
    name: str
    async def ainvoke(self, input: Dict[str, Any]) -> Any: ...


class ModelHubClient:
    def __init__(self, get_model_tool: AsyncTool, get_by_path_tool: AsyncTool):
        self._get_model = get_model_tool
        self._get_by_path = get_by_path_tool

    async def get_model(self, model_id: str) -> Dict[str, Any]:
        return await self._get_model.ainvoke({"model_id": model_id})

    async def get_by_path(self, file_path: str) -> Dict[str, Any]:
        return await self._get_by_path.ainvoke({"file_path": file_path})


class SnpeRegistryClient:
    def __init__(self, list_tools: AsyncTool, get_tool_spec: AsyncTool, run_tool: AsyncTool):
        self._list_tools = list_tools
        self._get_tool_spec = get_tool_spec
        self._run_tool = run_tool

    async def list_tools(self) -> List[Dict[str, Any]]:
        return await self._list_tools.ainvoke({})

    async def get_tool_spec(self, tool_name: str, view: str) -> Dict[str, Any]:
        return await self._get_tool_spec.ainvoke({"tool_name": tool_name, "view": view})

    async def run_tool(self, tool_name: str, args: Dict[str, Any], workspace_root: str) -> Dict[str, Any]:
        return await self._run_tool.ainvoke({"tool_name": tool_name, "args": args, "workspace_root": workspace_root})



--------////

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from pydantic import ValidationError

from .schemas import ConversionContext, ConversionPlanIR
from .prompts import CONVERTER_PLANNER_SYSTEM


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robust-ish JSON extraction:
    - if model returns pure JSON, json.loads works
    - otherwise, find first '{' and last '}' and parse
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Planner did not return JSON")
        return json.loads(text[start:end+1])


class ConverterPlanner:
    """
    ReAct planner: LLM can call list_tools/get_tool_spec/modelhub tools to decide the plan.
    """

    def __init__(self, llm_with_tools, tools_by_name: Dict[str, Any], max_iters: int = 8):
        self.llm = llm_with_tools
        self.tools_by_name = tools_by_name
        self.max_iters = max_iters

    async def plan(self, ctx: ConversionContext) -> ConversionPlanIR:
        messages: List[Any] = [
            SystemMessage(content=CONVERTER_PLANNER_SYSTEM),
            HumanMessage(content=json.dumps(ctx.model_dump(), indent=2)),
        ]

        for _ in range(self.max_iters):
            ai: AIMessage = await self.llm.ainvoke(messages)
            messages.append(ai)

            # Tool calls?
            tool_calls = getattr(ai, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("args") or {}
                    if name not in self.tools_by_name:
                        # respond with tool error message
                        messages.append(ToolMessage(content=f"ERROR: unknown tool {name}", tool_call_id=tc.get("id")))
                        continue
                    tool = self.tools_by_name[name]
                    result = await tool.ainvoke(args)
                    messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc.get("id")))
                continue

            # No tool calls -> should be final JSON plan
            payload = _extract_json_object(ai.content or "")
            try:
                return ConversionPlanIR.model_validate(payload)
            except ValidationError as e:
                # Feed error back and ask to re-emit valid JSON
                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous output was not valid ConversionPlanIR JSON.\n"
                            f"Validation error:\n{e}\n\n"
                            "Re-emit ONLY valid ConversionPlanIR JSON."
                        )
                    )
                )

        raise RuntimeError("Planner exceeded max tool iterations without producing a valid plan")


-----///


from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .schemas import ConversionPlanIR, CompiledPlan, CompiledStep, ToolCatalogEntry
from .tool_facades import SnpeRegistryClient
from .config import ConverterPolicyConfig


class CompileError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _is_safe_relpath(p: str) -> bool:
    if not p or os.path.isabs(p):
        return False
    norm = os.path.normpath(p)
    return not (norm.startswith("..") or "/../" in norm.replace("\\", "/"))


def _resolve_selector(catalog: List[Dict[str, Any]], selector: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically match a tool from list_tools() output.
    """
    matches = []
    for t in catalog:
        if t.get("category") != selector.get("category"):
            continue
        if selector.get("source") and t.get("source") != selector.get("source"):
            continue
        if selector.get("target") and t.get("target") != selector.get("target"):
            continue
        if selector.get("accelerator") and t.get("accelerator") != selector.get("accelerator"):
            continue
        if selector.get("sdk") and t.get("sdk") != selector.get("sdk"):
            continue

        tags_all = selector.get("tags_all") or []
        tags_any = selector.get("tags_any") or []
        tool_tags = set(t.get("tags") or [])

        if tags_all and not set(tags_all).issubset(tool_tags):
            continue
        if tags_any and not set(tags_any).intersection(tool_tags):
            continue

        matches.append(t)

    if not matches:
        raise CompileError("NO_TOOL_MATCH", f"Selector {selector} matched 0 tools")
    if len(matches) > 1:
        raise CompileError("AMBIGUOUS_TOOL_MATCH", f"Selector {selector} matched multiple tools: {[m.get('name') for m in matches]}")
    return matches[0]


def _spec_arg_names(spec: Dict[str, Any]) -> List[str]:
    return [a.get("name") for a in (spec.get("args") or []) if a.get("name")]


def _map_required_args(required_spec: Dict[str, Any], abs_inputs: Dict[str, str], abs_outputs: Dict[str, str]) -> Dict[str, Any]:
    """
    Conservative mapping: fill required I/O args based on name patterns.
    If your tool naming conventions differ, update this function.
    """
    args = {}
    for a in (required_spec.get("args") or []):
        name = a.get("name")
        if not name:
            continue
        n = name.lower()

        if ("input" in n and "model" in n) or name in ("input_model", "model", "input"):
            # Prefer 'model' input key if present
            if "model" in abs_inputs:
                args[name] = abs_inputs["model"]
            else:
                # fallback: first input
                args[name] = next(iter(abs_inputs.values()))
            continue

        if ("output" in n and ("dlc" in n or "model" in n)) or name in ("output_dlc", "output_model", "out"):
            if "dlc" in abs_outputs:
                args[name] = abs_outputs["dlc"]
            else:
                args[name] = next(iter(abs_outputs.values()))
            continue

        # Refuse to guess non-I/O required args
        raise CompileError("REQUIRED_ARG_UNMAPPED", f"Unhandled required arg '{name}' (update compiler mapping)")
    return args


class PlanCompiler:
    """
    Deterministically compiles ConversionPlanIR into CompiledPlan:
    - resolves selectors -> tool_name
    - validates tool specs
    - expands policy bucket to optional args
    - sandboxes IO paths
    """

    def __init__(self, snpe: SnpeRegistryClient, policy: ConverterPolicyConfig):
        self.snpe = snpe
        self.policy = policy

    async def compile(
        self,
        *,
        plan_ir: ConversionPlanIR,
        experiment_id: str,
        workspace_root: str,
        tool_catalog_snapshot: List[Dict[str, Any]],
    ) -> CompiledPlan:
        steps: List[CompiledStep] = []

        # Validate IR relpaths
        for s in plan_ir.steps:
            for p in list(s.io.inputs.values()) + list(s.io.outputs.values()):
                if not _is_safe_relpath(p):
                    raise CompileError("PATH_ESCAPE", f"Unsafe workspace-relative path: {p}")

        runtime = plan_ir.intent.target_runtime

        for step in plan_ir.steps:
            selector = step.tool_selector.model_dump()
            tool_entry = _resolve_selector(tool_catalog_snapshot, selector)
            tool_name = tool_entry["name"]

            # derive absolute IO
            abs_inputs = {k: os.path.join(workspace_root, v) for k, v in step.io.inputs.items()}
            abs_outputs = {k: os.path.join(workspace_root, v) for k, v in step.io.outputs.items()}

            # fetch specs
            req_spec = await self.snpe.get_tool_spec(tool_name, "required")
            opt_spec = await self.snpe.get_tool_spec(tool_name, "optional")

            req_names = set(_spec_arg_names(req_spec))
            opt_names = set(_spec_arg_names(opt_spec))

            # required args mapping
            required_args = _map_required_args(req_spec, abs_inputs, abs_outputs)

            # optional args from: intent + policy bucket + step params (high-level only)
            bucket = step.policy_bucket or plan_ir.intent.model_dump().get("policy_bucket")  # not present but safe
            bucket = bucket or "balanced"

            policy_args = self.policy.buckets.get(runtime, {}).get(step.op, {}).get(bucket, {})
            # intent-level allowed knobs
            intent_args = {}
            if "precision" in opt_names:
                intent_args["precision"] = plan_ir.intent.precision
            if plan_ir.intent.quantization != "none" and "quantization" in opt_names:
                intent_args["quantization"] = plan_ir.intent.quantization

            # step params (still validated against opt spec)
            step_params = {}
            for k, v in (step.params or {}).items():
                if k in opt_names:
                    step_params[k] = v

            optional_args = {}
            for src in (policy_args, intent_args, step_params):
                for k, v in src.items():
                    if k in opt_names:
                        optional_args[k] = v

            # final args validation: required keys must exist
            for k in required_args.keys():
                if k not in req_names:
                    raise CompileError("REQ_ARG_NOT_IN_SPEC", f"Required arg '{k}' not in required spec for {tool_name}")

            # optional keys must exist
            unknown_opt = [k for k in optional_args.keys() if k not in opt_names]
            if unknown_opt:
                raise CompileError("ARG_NOT_SUPPORTED", f"Optional args not supported by {tool_name}: {unknown_opt}")

            expected_outputs = list(abs_outputs.values())
            validations = [{"type": "file_exists", "path": p} for p in expected_outputs]

            steps.append(
                CompiledStep(
                    step_id=step.step_id,
                    op=step.op,
                    tool_name=tool_name,
                    tool_category=tool_entry.get("category", "unknown"),
                    args={**required_args, **optional_args},
                    abs_inputs=abs_inputs,
                    abs_outputs=abs_outputs,
                    depends_on=step.depends_on,
                    expected_outputs=expected_outputs,
                    validations=validations,
                )
            )

        return CompiledPlan(
            experiment_id=experiment_id,
            workspace_root=workspace_root,
            intent=plan_ir.intent.model_dump(),
            tool_catalog_snapshot=tool_catalog_snapshot,
            steps=steps,
            policy_version=self.policy.policy_version,
            compiler_metadata={"compiled_from_plan_version": plan_ir.plan_version},
        )



------////


from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from .schemas import CompiledPlan, ConversionResult
from .tool_facades import ModelHubClient, SnpeRegistryClient


class ExperimentService(Protocol):
    def create(self, **kwargs) -> None: ...
    def update_status(self, experiment_id: str, status: str, error: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None: ...
    def add_artifacts(self, experiment_id: str, artifacts: List[Dict[str, str]]) -> None: ...


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _relpath(root: str, abs_path: str) -> str:
    return os.path.relpath(abs_path, root)


def _validate_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"Validation failed: file not found: {path}")


class PlanExecutor:
    """
    Deterministic execution:
    - resolve model
    - create experiment + workspace
    - execute compiled steps sequentially using run_tool
    - persist logs + artifacts + statuses
    """

    def __init__(
        self,
        *,
        modelhub: ModelHubClient,
        snpe: SnpeRegistryClient,
        experiments: ExperimentService,
        experiments_root: str,
    ):
        self.modelhub = modelhub
        self.snpe = snpe
        self.experiments = experiments
        self.experiments_root = experiments_root

    async def resolve_model(self, model_ref: Dict[str, str]) -> Dict[str, Any]:
        if model_ref["type"] == "modelhub_id":
            return await self.modelhub.get_model(model_ref["value"])
        return await self.modelhub.get_by_path(model_ref["value"])

    def create_workspace(self, user: str, experiment_id: str) -> Dict[str, str]:
        root = os.path.join(self.experiments_root, user, experiment_id)
        paths = {
            "root": root,
            "input": os.path.join(root, "input"),
            "converted": os.path.join(root, "converted"),
            "logs": os.path.join(root, "logs"),
        }
        for p in paths.values():
            _ensure_dir(p)
        return paths

    def copy_model_to_workspace(self, model: Dict[str, Any], ws: Dict[str, str]) -> str:
        src = model["absolute_path"]
        dst = os.path.join(ws["input"], model["filename"])
        shutil.copy2(src, dst)
        return dst

    async def run(self, *, user: str, experiment_id: str, compiled: CompiledPlan, model: Dict[str, Any]) -> ConversionResult:
        ws_root = compiled.workspace_root
        logs_dir = os.path.join(ws_root, "logs")
        _ensure_dir(logs_dir)

        self.experiments.update_status(experiment_id, "EXECUTING")

        # Execute steps in order (compiler already set)
        for step in compiled.steps:
            step_log_prefix = os.path.join(logs_dir, f"{step.step_id}")

            # Idempotence: skip if outputs exist
            if step.skip_if_outputs_exist and step.expected_outputs and all(os.path.exists(p) for p in step.expected_outputs):
                _write_text(f"{step_log_prefix}.skipped.txt", "Skipped: outputs already exist\n")
                continue

            self.experiments.update_status(experiment_id, f"STEP_RUNNING:{step.step_id}")

            exec_result = await self.snpe.run_tool(
                tool_name=step.tool_name,
                args=step.args,
                workspace_root=ws_root,
            )

            _write_json(f"{step_log_prefix}.exec_result.json", exec_result)
            _write_text(f"{step_log_prefix}.stdout.txt", str(exec_result.get("stdout", "")))
            _write_text(f"{step_log_prefix}.stderr.txt", str(exec_result.get("stderr", "")))

            exit_code = exec_result.get("exit_code", 0)
            if exit_code not in (0, "0", None):
                raise RuntimeError(f"Step {step.step_id} failed (exit_code={exit_code}). Check logs.")

            # Validations
            for v in step.validations:
                if v.get("type") == "file_exists":
                    _validate_file_exists(v["path"])

        # Artifact discovery: simple baseline (dlc if exists)
        artifacts = []
        dlc_candidate = None
        for st in compiled.steps:
            for outp in st.abs_outputs.values():
                if outp.endswith(".dlc") and os.path.exists(outp):
                    dlc_candidate = outp
                    break

        if dlc_candidate:
            artifacts.append({"type": "dlc", "path": _relpath(ws_root, dlc_candidate)})

        self.experiments.add_artifacts(experiment_id, artifacts)
        self.experiments.update_status(experiment_id, "COMPLETED")

        return ConversionResult(experiment_id=experiment_id, status="COMPLETED", artifacts=artifacts)

    async def run_safe(self, *, user: str, experiment_id: str, compiled: CompiledPlan, model: Dict[str, Any]) -> ConversionResult:
        try:
            return await self.run(user=user, experiment_id=experiment_id, compiled=compiled, model=model)
        except Exception as e:
            # Best-effort error handling
            ws_root = compiled.workspace_root
            logs_dir = os.path.join(ws_root, "logs")
            _ensure_dir(logs_dir)
            _write_text(os.path.join(logs_dir, "failure.txt"), str(e))
            self.experiments.update_status(experiment_id, "FAILED", error=str(e))
            return ConversionResult(experiment_id=experiment_id, status="FAILED", error=str(e))




-----///


from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from agents.converter_v2.config import ConverterConfig
from agents.converter_v2.schemas import ConversionContext, ConversionPlanIR, ConversionResult, ResolvedModelInfo
from agents.converter_v2.tool_facades import ModelHubClient, SnpeRegistryClient
from agents.converter_v2.planner import ConverterPlanner
from agents.converter_v2.compiler import PlanCompiler, CompileError
from agents.converter_v2.executor import PlanExecutor


class ConverterAgentV2:
    """
    Convert node for main orchestrator.

    Inputs expected in main GraphState:
      - state["conversion_request"] or state["conversion_intent"] (dict-like):
          { model_ref: {...}, target_runtime, target_soc, precision? ... }
      - state["session_id"] (recommended)
      - state["user"] (optional)
      - state["recent_chat_summary"] (optional)

    Output:
      - state["conversion_result"] : dict
      - state["last_experiment_id"]: str
    """

    def __init__(
        self,
        *,
        config: ConverterConfig,
        llm_planner_with_tools,
        tools_by_name: Dict[str, Any],
        modelhub: ModelHubClient,
        snpe: SnpeRegistryClient,
        experiments: Any,
        user_provider: Optional[Any] = None,
    ):
        self.config = config
        self.tools_by_name = tools_by_name

        # Planner gets only specific tools (you should pass only those in llm binding)
        self.planner = ConverterPlanner(llm_planner_with_tools, tools_by_name, max_iters=config.planner_max_iters)

        # Compiler (deterministic)
        self.compiler = PlanCompiler(snpe=snpe, policy=config.policy)

        # Executor (deterministic)
        self.executor = PlanExecutor(
            modelhub=modelhub,
            snpe=snpe,
            experiments=experiments,
            experiments_root=config.paths.experiments_root,
        )

        self.experiments = experiments
        self.user_provider = user_provider
        self.modelhub = modelhub
        self.snpe = snpe

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # 0) user/session
        user = None
        if self.user_provider:
            user = self.user_provider(state)
        user = user or state.get("user") or "default_user"

        session_id = state.get("session_id") or state.get("thread_id") or "session"

        # 1) Extract request (you can unify naming)
        req = state.get("conversion_request") or state.get("conversion_intent")
        if not req:
            raise RuntimeError("Missing conversion_request/conversion_intent in main state")

        # must include model_ref
        model_ref = req.get("model_ref") or {}
        if "type" not in model_ref or "value" not in model_ref:
            raise RuntimeError("conversion_request.model_ref must include {type,value}")

        latest_user_message = state.get("latest_user_message") or state.get("last_user_message") or ""
        recent_summary = state.get("recent_chat_summary")

        # 2) Deterministic: resolve model + create experiment + workspace
        experiment_id = str(uuid.uuid4())
        ws_paths = self.executor.create_workspace(user=user, experiment_id=experiment_id)
        ws_root = ws_paths["root"]

        # Resolve model deterministically (so planner has correct metadata)
        model = await self.executor.resolve_model(model_ref)
        for k in ("id", "filename", "absolute_path"):
            if k not in model:
                raise RuntimeError(f"ModelHub model missing '{k}'. Keys: {list(model.keys())}")

        copied_model_abs = self.executor.copy_model_to_workspace(model, ws_paths)

        # DB record early
        self.experiments.create(
            experiment_id=experiment_id,
            user=user,
            model_id=model["id"],
            target_runtime=req.get("target_runtime", "snpe"),
            target_soc=req.get("target_soc", "unknown"),
            status="CREATED",
            created_at=datetime.utcnow(),
            metadata={"planner": "react", "version": "v2"},
            params=req,
        )
        self.experiments.update_status(experiment_id, "MATERIALIZED")

        # 3) Deterministic: get tool catalog snapshot (lightweight)
        tool_catalog = await self.snpe.list_tools()

        # 4) Build planner context
        ctx = ConversionContext(
            session_id=session_id,
            user=user,
            latest_user_message=latest_user_message,
            recent_chat_summary=recent_summary,
            model_ref=model_ref,
            resolved_model=ResolvedModelInfo(
                model_id=model["id"],
                filename=model["filename"],
                absolute_path=model["absolute_path"],
            ),
            tool_catalog=tool_catalog,
            previous_experiments=[],
            default_policy_bucket="balanced",
            constraints={"allow_experimental": False},
            hints={"workspace_input_model": os.path.relpath(copied_model_abs, ws_root)},
        )

        # 5) Planner (LLM ReAct) -> Plan IR
        try:
            plan_ir: ConversionPlanIR = await self.planner.plan(ctx)
        except Exception as e:
            self.experiments.update_status(experiment_id, "FAILED", error=f"Planner failed: {e}")
            state["conversion_result"] = ConversionResult(experiment_id=experiment_id, status="FAILED", error=str(e)).model_dump()
            state["last_experiment_id"] = experiment_id
            return state

        # NEEDS_INPUT -> return to chat
        if plan_ir.status == "NEEDS_INPUT":
            self.experiments.update_status(experiment_id, "NEEDS_INPUT", extra={"questions": plan_ir.needs_input.model_dump() if plan_ir.needs_input else None})
            state["conversion_result"] = ConversionResult(
                experiment_id=experiment_id,
                status="NEEDS_INPUT",
                needs_input=plan_ir.needs_input,
            ).model_dump()
            state["last_experiment_id"] = experiment_id
            return state

        # 6) Compile plan deterministically
        try:
            compiled = await self.compiler.compile(
                plan_ir=plan_ir,
                experiment_id=experiment_id,
                workspace_root=ws_root,
                tool_catalog_snapshot=tool_catalog,
            )
        except CompileError as ce:
            self.experiments.update_status(experiment_id, "FAILED", error=str(ce), extra={"code": ce.code})
            state["conversion_result"] = ConversionResult(experiment_id=experiment_id, status="FAILED", error=str(ce)).model_dump()
            state["last_experiment_id"] = experiment_id
            return state

        # Persist plan artifacts
        with open(os.path.join(ws_root, "plan_ir.json"), "w", encoding="utf-8") as f:
            json.dump(plan_ir.model_dump(), f, indent=2)
        with open(os.path.join(ws_root, "compiled_plan.json"), "w", encoding="utf-8") as f:
            json.dump(compiled.model_dump(), f, indent=2)

        # 7) Execute compiled plan deterministically
        result = await self.executor.run_safe(user=user, experiment_id=experiment_id, compiled=compiled, model=model)

        # Copy back to main state
        state["conversion_result"] = result.model_dump()
        state["last_experiment_id"] = experiment_id
        return state



------////

# ConverterAgent V2 (Planner ReAct + Compiler + Executor)

This implementation follows **Option A (state-of-the-art)**:

1) **Planner (LLM, ReAct)** chooses a semantic plan using tool introspection:
   - list_tools
   - get_tool_spec
   - modelhub_get_model / get_by_path (optional)

2) **Compiler (deterministic)** validates and compiles the plan:
   - resolves tool selectors to exact tool names
   - expands policy buckets to optional args (validated against tool spec)
   - sandboxes IO paths within workspace root

3) **Executor (deterministic)** runs the compiled steps:
   - sequential run_tool calls
   - idempotence optional
   - logs + artifacts + experiment status

The executor never “reasons”; it only runs a compiled build plan.

## Integration points

You must wire:
- ModelHub MCP tools: get_model, get_by_path
- SNPE/QNN MCP tools: list_tools, get_tool_spec, run_tool
- ExperimentService: create/update_status/add_artifacts

You must also bind ONLY the planner tools to the planner LLM.

## Files written per experiment
- input/ (copied source model)
- plan_ir.json
- compiled_plan.json
- logs/
- converted/ (artifacts)



------//

You are ConverterPlanner.

Goal:
Produce a ConversionPlanIR to convert a model to the requested target runtime and SoC.

Allowed tools:
- list_tools
- get_tool_spec
- modelhub_get_model / modelhub_get_by_path (optional)

Rules:
- Never call run_tool.
- Do not output CLI flags; choose high-level params and policy_bucket.
- Use tool_selector (category/source/target/accelerator/tags) instead of hardcoding tool names.
- Use workspace-relative IO paths only (no absolute paths, no ..).
- If required info is missing, return NEEDS_INPUT with questions.

Output:
Return ONLY a valid ConversionPlanIR JSON object.



----//





# ConverterAgent

ConverterAgent is a **deterministic execution sub-graph** responsible for converting deep learning models into target runtimes (SNPE / QNN / etc.) inside the *Agentic DL Workflow Suite*.

It is intentionally **non-conversational**, **reproducible**, and **auditable**.

---

## 1) What it does

ConverterAgent performs:

1. **Resolve model** (from ModelHub via MCP; ModelHub is read-only)
2. **Create experiment record** (DB)
3. **Materialize workspace** (create dirs + copy model into `input/`)
4. **Select conversion tool** using reflective MCP registry (`list_tools`)
5. **Resolve required args** via `get_tool_spec(view="required")` (I/O mapping only)
6. **Apply optional args** only from explicit intent/policy (`precision`, `quantization`, etc.)
7. **Run conversion** via MCP `run_tool`
8. **Persist logs + artifacts** and mark experiment `COMPLETED` (or `FAILED`)

ConverterAgent **does not**:
- Participate in chat loops
- Use chat history as context
- Reason about CLI flags
- Modify ModelHub
- Perform profiling or patch-search

---

## 2) Design principles

**Execution plane vs reasoning plane**

ConverterAgent lives in the **execution plane**:
- deterministic
- restartable
- reproducible
- audit-friendly

Agentic “intelligence” (iteration, argument deltas, choosing next experiments) belongs outside:
- ChatAgent / SupervisorAgent
- future ExperimentReasonerAgent

**Reflective tool registry**
SNPE/QNN tools can have huge schemas. To avoid context bloat, the SNPE/QNN MCP server exposes:
- `list_tools` (lightweight list, no args)
- `get_tool_spec` (fetch required/optional args only when needed)
- `run_tool` (execute)

---

## 3) Inputs

ConverterAgent consumes a `ConversionIntent` object.

Example:

```json
{
  "model_ref": { "type": "modelhub_id", "value": "resnet50_tf" },
  "source_framework": "tensorflow",
  "target_runtime": "snpe",
  "target_soc": "sm8550",
  "precision": "fp16",
  "quantization": "none"
}

Notes:
	•	No CLI flags in intent.
	•	Only “intent-level” fields (source/target/runtime/soc/precision/quantization).

⸻

4) Outputs

ConverterAgent returns a compact ConversionResult and writes artifacts to the workspace.

Result injected back into the main graph state

{
  "experiment_id": "uuid",
  "status": "COMPLETED",
  "artifacts": [
    { "type": "dlc", "path": "converted/model.dlc" }
  ],
  "error": null
}

Workspace layout

/data/experiments/<user>/<experiment_id>/
  ├── input/
  │   └── <model-file>
  ├── converted/
  │   └── model.dlc
  ├── logs/
  │   ├── conversion.stdout.txt
  │   ├── conversion.stderr.txt
  │   ├── conversion.exec_result.json
  │   └── failure.txt (only on error)
  ├── metadata.json
  └── params.json


⸻

5) Sub-graph flow (LangGraph)

ConverterAgent is implemented as a LangGraph sub-graph:

resolve_model
→ create_experiment
→ materialize_workspace
→ select_tool
→ resolve_tool_plan
→ run_conversion
→ END

All failures route to:

handle_failure → END


⸻

6) MCP tool expectations

ModelHub (read-only)

Required MCP tools:
	•	modelhub_get_model(model_id)
	•	modelhub_get_by_path(file_path)

Expected response fields:
	•	id
	•	filename
	•	absolute_path

SNPE/QNN reflective registry (stateless executor)

Required MCP tools:
	•	list_tools()
	•	get_tool_spec(tool_name, view) where view ∈ {required, optional, all}
	•	run_tool(tool_name, args, workspace_root)

list_tools() should return entries with at least:
	•	name
	•	category (must include "conversion")
	•	source (e.g. tensorflow)
	•	target (e.g. snpe)

⸻

7) Determinism rules (MUST)

ConverterAgent MUST:
	•	Fail if no tool matches the intent
	•	Fail if multiple tools match the intent (ambiguity)
	•	Only resolve required args that are clearly I/O paths
	•	Never “guess” non-I/O required args
	•	Never write outside the workspace root
	•	Never modify ModelHub

⸻

8) Integration into main orchestrator graph

ConverterAgent is plugged into the main orchestrator as a single node:

START → supervisor → convert → END

It does not participate in:
	•	chat → tools → chat loop
	•	RAG loop

The main graph passes only:
	•	conversion_intent
	•	session_id / thread_id
	•	user (optional)

And receives:
	•	conversion_result
	•	last_experiment_id

⸻

9) Dependencies you must provide (integration points)

You must supply concrete implementations (DB/MCP wrappers) for:
	•	ModelHub client:
	•	get_model(model_id) -> dict
	•	get_by_path(file_path) -> dict
	•	SNPE/QNN registry client:
	•	list_tools() -> list[dict]
	•	get_tool_spec(tool_name, view) -> dict
	•	run_tool(tool_name, args, workspace_root) -> dict
	•	Experiment service:
	•	create(...)
	•	update_status(experiment_id, status, error=None, extra=None)
	•	add_artifacts(experiment_id, artifacts)

⸻

10) Common failure modes
	•	Tool schema mismatch (required arg names changed)
	•	Tool ambiguity (multiple conversion tools match)
	•	Missing output artifact after tool run
	•	Non-zero exit code or tool error reported in exec result
	•	Filesystem permission / disk space issues

All failures must mark the experiment as FAILED and write logs/failure.txt best-effort.

⸻


