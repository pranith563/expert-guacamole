# Converter (Planner → Tools → Planner → Compiler → Executor) — Detailed README

This document describes the **Converter** subsystem in the Agentic DL Workflow Suite.  
Converter is responsible for converting a selected model (from ModelHub) into a runtime-specific artifact (e.g., SNPE/QNN DLC + optional cached binaries) and recording the full run as an Experiment.

The design is **agentic where it helps** (planning) and **deterministic where it must** (compile/execute).

---

## 1) What Converter Does

Given a user request like:

> “Convert model X to SNPE DLC for sm8550, fp16.”

Converter will:

1. **Resolve the input model** (ModelHub MCP)
2. **Create an Experiment run** (Experiments MCP)
3. **Create a workspace** on disk and copy the model to `input/`
4. **Plan conversion steps** using an LLM ReAct planner (may consult QAIRT tool registry)
5. **Compile the plan** deterministically:
   - validate required args
   - validate optional args are supported
   - sanitize and expand workspace-relative paths into absolute paths
6. **Execute steps** deterministically via QAIRT MCP tool(s)
7. **Validate outputs**, attach artifacts to Experiment, update status
8. Return a **structured conversion_result** for Chat/UI presentation

---

## 2) Key Architectural Principles

### 2.1 Read-only ModelHub, write-only Workspace
- **ModelHub is immutable** (read-only)
- Every conversion run happens in a **new workspace** under experiments:

`/data/experiments/<user_id>/<run_id>/`

Nothing writes back into ModelHub.

### 2.2 Agentic planning, deterministic execution
- **Planner** is agentic (LLM + tool calls) to pick correct conversion tool + args.
- **Compiler** and **Executor** are deterministic for safety and reproducibility.

### 2.3 Tool use follows the main graph pattern
Planner uses **ToolNode via edges** like your main chat flow:
- `planner ↔ convert_tools`

Compiler/Executor call tools **directly** using `tool_manager.get_tool(...).ainvoke(...)`.

### 2.4 Loopback on errors
If compiler or executor encounters a recoverable error, the subgraph loops back:
- `compiler/executor → planner`

Bounded by `max_replans` to prevent infinite cycles.

### 2.5 Needs-input is explicit (no guessing)
Planner may return:
- `status = NEEDS_INPUT` with `questions[]`

Converter ends cleanly; Chat/Supervisor asks the user and then re-runs conversion with updated request.

---

## 3) Converter Subgraph

Converter runs as a **subgraph** invoked by `ConverterAgent` (main graph node).

### 3.1 Node layout

START
→ planner
→ (if tool_calls) convert_tools
→ planner
→ (if plan_ready) compiler
→ (if error & replans left) planner
→ executor
→ (if error & replans left) planner
→ END

### 3.2 Nodes and responsibilities

#### Planner (ReAct + ToolNode)
- Ensures deterministic initialization runs once (run + workspace + model copy)
- Maintains `state.messages` list (BaseMessage objects)
- Calls LLM once per tick
- If LLM outputs tool_calls → ToolNode runs tools → goes back to planner
- If LLM outputs plan JSON → sets `planner_done=True`

#### convert_tools (ToolNode)
- Executes tool calls created by planner
- Appends tool output messages into `state.messages`
- Always routes back to planner

#### Compiler (deterministic)
- Validates `plan` schema (pydantic)
- Fetches tool specs via `qairt_get_tool_spec` (required + optional)
- Ensures:
  - all required args are present
  - optional args are supported
- Converts workspace-relative paths in args to absolute paths safely (sandboxed join)
- Writes:
  - `plan.json`
  - `compiled_steps.json`
- On error:
  - sets `replan_needed=True`, `planner_feedback=...`
  - clears `compiled_steps`
  - routes back to planner (if replans left)

#### Executor (deterministic)
- Runs compiled steps sequentially via `qairt_run`
- Writes per-step execution logs
- Validates expected output files exist
- Attaches artifacts to Experiments
- Updates run status
- On error:
  - sets `replan_needed=True`, `planner_feedback=...`
  - routes back to planner (if replans left)

---

## 4) State Management

### 4.1 Separate `ConverterState`
Converter uses a dedicated `ConverterState` to avoid polluting global chat state.
Only minimal info is copied in/out.

**In from GraphState**
- `session_id` (or `thread_id`)
- `user_id`
- `conversion_request`
- optional `recent_chat_summary`

**Out to GraphState**
- `conversion_result`
- `last_experiment_id`

### 4.2 Key state fields

- `run_id`: Experiment run identifier
- `workspace_root`: absolute path to run workspace
- `workspace_input_model`: path relative to workspace (e.g., `input/model.onnx`)
- `messages`: list of LangChain `BaseMessage` objects for ReAct loop
- `plan`: raw planner JSON output
- `compiled_steps`: fully compiled steps ready for execution
- `replan_count`, `max_replans`: loopback guardrails
- `planner_feedback`: appended into messages when replanning
- `conversion_result`: final structured output for UI/chat

---

## 5) Tooling Dependencies

Converter uses MCP tools loaded into ToolManager catalog as namespaced keys:

### 5.1 ModelHub MCP tools
- `modelhub:modelhub_get_model`
- `modelhub:modelhub_get_by_path`

Used during deterministic init to resolve absolute model paths.

### 5.2 Experiments MCP tools
- `experiments:experiments_create_run`
- `experiments:experiments_update_run_status`
- `experiments:experiments_attach_artifacts`

Used to persist run state, status, and outputs.

### 5.3 QAIRT MCP tools
- `qairt:qairt_list_tools`
- `qairt:qairt_get_tool_spec`
- `qairt:qairt_run`

Used by planner (list/spec) and executor (run).

> NOTE: Tool tagging is currently server-level.
> Planner is prevented from executing conversion by prompt instruction only (for now).

---

## 6) Workspace Layout

Workspace is created on disk under:

`/data/experiments/<user_id>/<run_id>/`

Example:

/data/experiments/pranith/3c1a…/
input/
model.onnx
converted/
model.dlc
cached/
htp/
…
logs/
convert.exec_result.json
last_error.json
metadata.json
plan.json
compiled_steps.json

### Files explained
- `metadata.json`: resolved model info + run metadata
- `plan.json`: planner output (raw)
- `compiled_steps.json`: compiled plan after validation
- `logs/*.exec_result.json`: per-step execution results from QAIRT tools
- `logs/last_error.json`: last compiler/executor error (if any)

---

## 7) Planner Output Contract

Planner must output a single JSON object with:

### OK plan
```json
{
  "status": "OK",
  "intent": { "target_soc": "sm8550", "precision": "fp16" },
  "steps": [
    {
      "step_id": "convert",
      "tool_name": "snpe-tensorflow-to-dlc",
      "required_args": { "...": "..." },
      "optional_args": { "...": "..." },
      "expected_outputs": ["converted/model.dlc"],
      "depends_on": []
    }
  ],
  "rationale": "..."
}

Needs input plan

{
  "status": "NEEDS_INPUT",
  "questions": [
    "Which SoC are you targeting? (sm8550, sm8650, ...)",
    "Do you want fp16 or int8?"
  ],
  "intent": {},
  "steps": [],
  "rationale": "Missing required choices."
}

Path rule

All file/dir args emitted by planner must be workspace-relative (not absolute).

⸻

8) Error Handling & Loopback

8.1 When do we loop back?
	•	Compiler errors: missing required args, unsupported optional args, invalid path, invalid plan schema
	•	Executor errors: nonzero exit code, missing expected outputs

8.2 How loopback works
	•	Compiler/Executor sets:
	•	replan_needed=True
	•	planner_feedback describing the issue
	•	planner_feedback_consumed=False
	•	Graph routes back to planner if replan_count < max_replans

Planner injects feedback into messages exactly once and attempts a new plan.

8.3 Bounding

max_replans prevents infinite loops.

⸻

9) Needs-input Flow (asking the user)

If planner cannot safely determine critical fields (SoC, datatype, quantization, etc.), it must output NEEDS_INPUT.

Converter returns conversion_result.status = NEEDS_INPUT and includes questions[].

The Chat/Supervisor should:
	•	ask the user those questions
	•	update GraphState["conversion_request"]
	•	re-run conversion

⸻

10) Main Graph Integration

10.1 Main graph creates ToolManager and tool nodes
	•	main graph has a global ToolNode for chat (ToolNode(all_tools))
	•	converter subgraph creates its own ToolNode (ToolNode(tools_for_mode("convert")))

No cross-graph edges are required (and not supported).

10.2 Supervisor contract

Supervisor must set:
	•	state["active_task_type"] = "convert"
	•	state["conversion_request"] = {...}

⸻

11) Configuration

ConverterConfig controls:
	•	experiments_root (default: /data/experiments)
	•	tool_mode (default: convert)
	•	planner_max_tool_iters (default: 8)
	•	max_replans (default: 2)

Tool names are stored in ConverterToolNames:
	•	update here if namespaced keys change.

⸻

12) Extending Converter

12.1 Add caching steps

Planner can output multi-step plans:
	•	step1: convert
	•	step2: cache (depends_on convert)

Compiler/Executor already supports step sequences and dependencies (basic list order).

12.2 Add deterministic argument templates

For common use cases, compiler can apply:
	•	policy presets (fast / balanced / max_perf)
	•	validated optional args sets

12.3 Stronger tool policy

Later, add per-tool tagging or allowlist/denylist:
	•	separate planner-only QAIRT tools from executor tools

⸻

13) Debugging Checklist

If conversion fails:
	1.	Check conversion_result in GraphState
	2.	Open workspace:
	•	metadata.json (model path sanity)
	•	plan.json (planner output)
	•	compiled_steps.json (compiler decisions)
	•	logs/*.exec_result.json (tool outputs)
	•	logs/last_error.json (failure reason)

Common issues:
	•	ModelHub returned missing absolute_path key → adjust normalization
	•	QAIRT tool spec schema differs from expected → adjust compiler parsing
	•	Planner emitted absolute paths → compiler rejects via safe_join
	•	Expected output path wrong → planner needs to fix expected_outputs

⸻

14) Quick Reference: Required Tools

ModelHub
	•	modelhub:modelhub_get_model
	•	modelhub:modelhub_get_by_path

Experiments
	•	experiments:experiments_create_run
	•	experiments:experiments_update_run_status
	•	experiments:experiments_attach_artifacts

QAIRT
	•	qairt:qairt_get_tool_spec
	•	qairt:qairt_run
	•	(planner may call) qairt:qairt_list_tools

⸻

15) Converter Output (for UI / Chat)

Converter returns a structured object:

Completed

{
  "status": "COMPLETED",
  "run_id": "...",
  "workspace_root": "...",
  "artifacts": [
    {"type":"dlc","path":"converted/model.dlc"}
  ],
  "timestamp": "..."
}

Needs input

{
  "status": "NEEDS_INPUT",
  "run_id": "...",
  "questions": ["Which SoC?", "fp16 or int8?"],
  "timestamp": "..."
}

Failed

{
  "status": "FAILED",
  "run_id": "...",
  "error": {"code":"EXECUTOR_ERROR","message":"...","where":"executor"},
  "timestamp": "..."
}


⸻

16) Recommended Next Step

After converter is stable:
	1.	Add a small post-convert routing: convert → chat_agent to present results nicely
	2.	Reuse the same “planner/compiler/executor” pattern for ProfilerAgent
	3.	Add experiment “rerun with modifications” by reusing previous plan.json + overrides

