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


