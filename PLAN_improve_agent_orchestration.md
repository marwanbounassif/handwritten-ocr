# Plan: Modernize OCR Agent — Libraries & LangGraph

## Context

The agentic OCR pipeline is well-architected with clean separation of concerns. However, several components are hand-rolled where battle-tested libraries exist, and the orchestrator's state machine pattern maps naturally to LangGraph. This plan modernizes the stack while preserving what's already good.

### What's already good (preserve as-is)
- **Agent prompts** ([agents.py](ocr_agent/agents.py)) — well-crafted f-string templates with clear role definitions and OCR error patterns
- **Image preprocessing pipeline** ([tools.py:566-743](ocr_agent/tools.py#L566-L743)) — domain-specific registry of composable transforms
- **OCR model management** ([tools.py:746-842](ocr_agent/tools.py#L746-L842)) — transformers-native with memory management for GPU sharing
- **Merge/compare logic** ([tools.py:393-563](ocr_agent/tools.py#L393-L563)) — LCS-aligned majority vote, domain-specific
- **CLI** ([transcribe.py](ocr_agent/transcribe.py)) — argparse entry point, single/batch processing

### Why NOT LangChain
- Agents are "prompt-in, JSON-out" functions — not tool-using agents needing `AgentExecutor`/`Tool` abstractions
- `ChatPromptTemplate` is strictly more verbose than the existing f-string templates
- Adds ~15 transitive dependencies for zero functional gain
- The project's agents have fixed behavior (Critic always analyzes, Editor always fixes) — LangChain's agent loop adds overhead for nothing

---

## Changes

### 1. Replace hand-rolled LLM client with `ollama` Python package

**Why:** [tools.py:142-318](ocr_agent/tools.py#L142-L318) is ~175 lines of manual HTTP requests, SSE stream parsing, `<think>` tag handling, JSON extraction with brace-depth matching, and retry logic. The `ollama` Python package is the official Ollama client — local-first by design, zero telemetry, no data leaves the machine.

**What to do:**
- In [tools.py](ocr_agent/tools.py), replace `call_llm()`, `_call_llm_stream()` with `ollama.chat()`:
  ```python
  import ollama

  def call_llm(system_prompt: str, user_message: str, ...) -> str:
      messages = []
      if system_prompt:
          messages.append({"role": "system", "content": system_prompt})
      messages.append({"role": "user", "content": user_message})

      if should_stream:
          chunks = []
          for part in ollama.chat(model=config.OPENAI_MODEL, messages=messages,
                                   stream=True, options={"temperature": temperature}):
              token = part["message"]["content"]
              chunks.append(token)
              sys.stdout.write(token)
              sys.stdout.flush()
          text = "".join(chunks)
      else:
          response = ollama.chat(model=config.OPENAI_MODEL, messages=messages,
                                  options={"temperature": temperature})
          text = response["message"]["content"]

      if config.LLM_ENABLE_THINKING:
          text = _strip_think_tags(text)
      return text
  ```
- For `call_llm_json()`, use `ollama.chat(format=PydanticModel.model_json_schema())` for structured output. Keep `parse_json_response()` as a fallback for models that don't support format constraints.
- Keep `_strip_think_tags()` — Qwen3-specific, no client library handles this.
- Remove `import requests` from tools.py.
- In [config.py](ocr_agent/config.py), rename `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL` to `OLLAMA_HOST`/`OLLAMA_MODEL` (the ollama package uses `OLLAMA_HOST` env var automatically, defaulting to `http://localhost:11434`).
- In [pyproject.toml](pyproject.toml): Add `ollama>=0.4`, remove `requests`.

**Files changed:** [tools.py](ocr_agent/tools.py), [config.py](ocr_agent/config.py), [pyproject.toml](pyproject.toml)

---

### 2. Replace custom CER/WER with `jiwer`

**Why:** [tools.py:69-139](ocr_agent/tools.py#L69-L139) is ~75 lines of hand-rolled Levenshtein/CER/WER. `jiwer` is the standard library in OCR/ASR research — uses `rapidfuzz` C extension (faster), handles edge cases, and makes evaluation results directly comparable to published benchmarks.

**What to do:**
- In [tools.py](ocr_agent/tools.py), replace `cer()`, `wer()`, `tier1_metrics()`:
  ```python
  import jiwer

  def cer(ground_truth: str, ocr_output: str, lower: bool = False) -> float:
      gt = normalize_text(ground_truth, lower)
      ocr = normalize_text(ocr_output, lower)
      if not gt:
          return 0.0 if not ocr else 1.0
      return jiwer.cer(gt, ocr)

  def wer(ground_truth: str, ocr_output: str, lower: bool = False) -> float:
      gt = normalize_text(ground_truth, lower)
      ocr = normalize_text(ocr_output, lower)
      if not gt:
          return 0.0 if not ocr else 1.0
      return jiwer.wer(gt, ocr)

  def tier1_metrics(ground_truth: str, ocr_output: str, lower: bool = False) -> dict:
      gt = normalize_text(ground_truth, lower)
      ocr = normalize_text(ocr_output, lower)
      return {
          "input": ocr_output,
          "cer": round(jiwer.cer(gt, ocr), 4) if gt else 0.0,
          "wer": round(jiwer.wer(gt, ocr), 4) if gt else 0.0,
          "exact_match": gt == ocr,
          "gt_chars": len(gt),
          "ocr_chars": len(ocr),
      }
  ```
- Keep `normalize_text()` — domain-specific curly quote/dash normalization that `jiwer` doesn't do.
- Keep `levenshtein()` and `_levenshtein_words()` — still used by `compare_versions()` and `merge_versions()` for text alignment (not evaluation). Consider swapping to `rapidfuzz.distance.Levenshtein` for speed.
- Remove the confusing `wer_token` field from `tier1_metrics()` — `jiwer.wer()` is the standard computation.
- In [pyproject.toml](pyproject.toml): Add `jiwer`.

**Files changed:** [tools.py](ocr_agent/tools.py), [pyproject.toml](pyproject.toml)

---

### 3. Add Pydantic models for agent output schemas

**Why:** Each agent in [agents.py](ocr_agent/agents.py) uses `.setdefault()` to handle missing fields (lines 92-96, 175-178, 246-250). This silently masks malformed LLM output. Pydantic gives: validation with meaningful errors, typed attribute access, IDE autocomplete, and self-documenting schemas. The Pydantic models also integrate with `ollama.chat(format=...)` for structured output.

**What to do:**
- In [agents.py](ocr_agent/agents.py), add these models:
  ```python
  from pydantic import BaseModel, Field, ConfigDict
  from typing import Literal

  class CriticIssue(BaseModel):
      model_config = ConfigDict(extra="ignore")
      description: str = ""
      severity: Literal["critical", "minor", "cosmetic"] = "minor"
      suggestion: str = ""

  class CriticSegment(BaseModel):
      model_config = ConfigDict(extra="ignore")
      text: str = ""
      confidence: int = Field(default=50, ge=0, le=100)
      issues: list[CriticIssue] = []

  class CriticResult(BaseModel):
      model_config = ConfigDict(extra="ignore")
      overall_confidence: int = Field(default=0, ge=0, le=100)
      segments: list[CriticSegment] = []
      verdict: Literal["accept", "needs_editing", "needs_reocr"] = "needs_editing"
      reasoning: str = ""

  class EditorChange(BaseModel):
      model_config = ConfigDict(extra="ignore")
      original: str = ""
      corrected: str = ""
      reason: str = ""
      confidence: int = Field(default=50, ge=0, le=100)

  class EditorResult(BaseModel):
      model_config = ConfigDict(extra="ignore")
      corrected_text: str
      changes: list[EditorChange] = []
      unresolved: list[str] = []

  class ArbitratorDecision(BaseModel):
      model_config = ConfigDict(extra="ignore")
      segment: str = ""
      chosen_version: int = 1
      reason: str = ""

  class ArbitratorResult(BaseModel):
      model_config = ConfigDict(extra="ignore")
      final_text: str
      decisions: list[ArbitratorDecision] = []
      confidence: int = Field(default=0, ge=0, le=100)
      uncertain_segments: list[str] = []
  ```
- Update each agent function to validate and return typed models:
  ```python
  def run_critic(transcription, previous_critique=None) -> CriticResult:
      # ... build user_msg same as before ...
      raw = call_llm_json(CRITIC_SYSTEM_PROMPT, user_msg)
      try:
          return CriticResult.model_validate(raw)
      except ValidationError as e:
          print(f"  [critic] WARNING: output validation failed: {e}")
          return CriticResult(overall_confidence=0, verdict="needs_editing",
                              reasoning="LLM output failed schema validation")
  ```
- In [orchestrator.py](ocr_agent/orchestrator.py), replace `.get("field", default)` dict access with attribute access:
  - `critique.get("overall_confidence", 0)` → `critique.overall_confidence`
  - `critique.get("verdict", "needs_editing")` → `critique.verdict`
  - `edit_result.get("corrected_text", ...)` → `edit_result.corrected_text`
  - `seg.get("issues", [])` → `seg.issues`
  - For `trace.log(full_output=critique)` → `trace.log(full_output=critique.model_dump())`
- In [pyproject.toml](pyproject.toml): Add `pydantic>=2.0`.

**Files changed:** [agents.py](ocr_agent/agents.py), [orchestrator.py](ocr_agent/orchestrator.py), [pyproject.toml](pyproject.toml)

---

### 4. Migrate orchestrator to LangGraph

**Why:** The orchestrator in [orchestrator.py](ocr_agent/orchestrator.py) is structurally a state graph: nodes (OCR, Critic, Editor, Arbitrator), conditional edges (accept/edit/reocr), and a loop. LangGraph makes this explicit, adds graph visualization, checkpointing for resumable runs, and streaming state updates. This is also a learning opportunity. LangGraph is standalone — it does NOT require LangChain.

**Graph structure:**
```
START → initial_ocr → critic → [routing]
                                  ├── accept → END
                                  ├── plateau → END
                                  ├── max_iterations → END
                                  ├── needs_reocr → reocr → critic (loop)
                                  └── needs_editing → editor → critic (loop)
```

**What to do:**

**4a. Define state as TypedDict** in a new file [ocr_agent/state.py](ocr_agent/state.py):
```python
from typing import TypedDict

class OCRState(TypedDict):
    image_path: str
    candidates: list[dict]
    critiques: list  # list[CriticResult] stored as dicts for serialization
    edits: list      # list[EditorResult] stored as dicts
    current_best: str
    current_score: float
    iteration: int
    max_iterations: int
    status: str      # "running" | "completed" | "max_iterations"
    reason: str      # "accept" | "plateau" | "exhausted" | "max_iterations" | ""
    strategies_used: list[str]
    plateau_count: int
    prev_score: float
    config: dict     # accept_threshold, plateau_patience, strategies list
```

**4b. Define node functions** — refactor the current `run_pipeline()` logic into separate functions. Each node receives `OCRState`, returns partial state updates:

- `node_initial_ocr(state)` — runs Phase 1: multi-strategy OCR reads, agreement check, merge candidates, unload OCR model. Extracted from current [orchestrator.py:37-93](ocr_agent/orchestrator.py#L37-L93).
- `node_critic(state)` — runs `run_critic()`, appends to `critiques`, updates `current_score`, handles plateau counting. Extracted from [orchestrator.py:104-144](ocr_agent/orchestrator.py#L104-L144).
- `node_editor(state)` — runs `run_editor()`, appends to `edits`, updates `current_best`. Extracted from [orchestrator.py:207-225](ocr_agent/orchestrator.py#L207-L225).
- `node_reocr(state)` — finds next unused strategy, runs OCR, calls arbitrator to merge with current best. Extracted from `_handle_reocr()` at [orchestrator.py:245-351](ocr_agent/orchestrator.py#L245-L351).
- `node_accept(state)` / `node_plateau(state)` / `node_max_iter(state)` — terminal nodes that set `status` and `reason`.

**4c. Define routing functions:**
```python
def route_after_critic(state: OCRState) -> str:
    latest = state["critiques"][-1]
    confidence = latest["overall_confidence"]
    verdict = latest["verdict"]

    if verdict == "accept" or confidence >= state["config"]["accept_threshold"]:
        return "accept"
    if state["plateau_count"] >= state["config"]["plateau_patience"]:
        return "plateau"
    if state["iteration"] >= state["max_iterations"]:
        return "max_iterations"
    if verdict == "needs_reocr":
        return "reocr"
    return "edit"

def route_after_reocr(state: OCRState) -> str:
    if state["reason"] == "exhausted":
        return "max_iterations"
    return "critic"
```

**4d. Build the graph** in [ocr_agent/graph.py](ocr_agent/graph.py):
```python
from langgraph.graph import StateGraph, START, END

def build_ocr_graph():
    builder = StateGraph(OCRState)

    builder.add_node("initial_ocr", node_initial_ocr)
    builder.add_node("critic", node_critic)
    builder.add_node("editor", node_editor)
    builder.add_node("reocr", node_reocr)
    builder.add_node("accept", node_accept)
    builder.add_node("plateau", node_plateau)
    builder.add_node("max_iterations", node_max_iter)

    builder.add_edge(START, "initial_ocr")
    builder.add_edge("initial_ocr", "critic")
    builder.add_conditional_edges("critic", route_after_critic, {
        "accept": "accept", "plateau": "plateau",
        "max_iterations": "max_iterations",
        "reocr": "reocr", "edit": "editor",
    })
    builder.add_edge("editor", "critic")
    builder.add_conditional_edges("reocr", route_after_reocr, {
        "critic": "critic", "max_iterations": "max_iterations",
    })
    builder.add_edge("accept", END)
    builder.add_edge("plateau", END)
    builder.add_edge("max_iterations", END)

    return builder.compile()
```

**4e. Update [transcribe.py](ocr_agent/transcribe.py)** to call `graph.invoke(initial_state)` instead of `run_pipeline()`.

**4f. Tracing integration:** LangGraph streams state updates via `graph.stream(state, stream_mode="updates")`. Integrate with existing [trace.py](ocr_agent/trace.py) by logging each streamed event. Alternatively, each node function can call `trace.log()` internally (simpler approach that preserves current trace format).

**4g. Keep [orchestrator.py](ocr_agent/orchestrator.py) temporarily** until the LangGraph version is verified, then delete it.

**Files changed:** New [ocr_agent/state.py](ocr_agent/state.py), new [ocr_agent/graph.py](ocr_agent/graph.py), [orchestrator.py](ocr_agent/orchestrator.py) (eventually removed), [transcribe.py](ocr_agent/transcribe.py), [pyproject.toml](pyproject.toml)

In [pyproject.toml](pyproject.toml): Add `langgraph`.

---

## Sequencing

Do these in order — each step should be a separate commit and verified independently:

1. **Pydantic models** — add schemas in agents.py, update orchestrator.py dict access. Foundation for steps 2-4.
2. **ollama package swap** — replace `requests`-based LLM client. Use Pydantic schemas with `format=` parameter. Zero behavioral change to agents.
3. **jiwer swap** — independent, safe, small.
4. **LangGraph orchestrator** — build new graph.py with nodes extracted from orchestrator.py, wire up in transcribe.py, verify against existing trace output, then remove old orchestrator.py.

## Verification

After each step:
- Run `poetry install` to confirm dependencies resolve
- Run `ocr data/input/test_notes.jpg` end-to-end and confirm the pipeline completes
- Compare output transcription against existing results in `data/output/` for behavioral regression
- For jiwer: compare `tier1_metrics()` output against previous values (small delta acceptable)
- For LangGraph: compare full trace JSON (iterations, verdicts, confidence scores) against previous trace to verify identical decision-making
