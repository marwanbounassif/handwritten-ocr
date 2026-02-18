# Plan: Scholar Council + RAG Context Store

## Context

OCR output is mostly clean after preprocessing improvements, but the pipeline still produces **semantic near-misses** — both readings are valid English but the meaning differs (e.g., "reported" vs "represented", "other schools" vs "that school"). The current single critic only flags corruption and accepts any linguistically valid sentence. We need a panel of agents that reason about **semantic plausibility** from different angles, and use **accumulated context** to disambiguate.

Two enhancements:
1. **Scholar Council** — a panel of 5 scholars that **replaces** the critic. Each scholar independently does the full analysis: identify issues (including ambiguities), propose readings, and vote on the verdict. Consensus drives routing.
2. **RAG Context Store** — TBD. Will provide accumulated context to the council.

## Graph — Council Replaces Critic, Same Topology

```
START -> initial_ocr -> [TBD: context retrieval] -> council <-> editor
                                                       |
                                                     reocr ──┐
                                                       |      |
                                                    council <─┘
                                                       |
                                             accept / plateau / max_iter
                                                       |
                                                  [TBD: store result] -> END
```

Routing outputs are identical to today: `accept`, `needs_editing`, `needs_reocr`, `plateau`, `max_iterations`. The `council` node is a drop-in replacement for `critic`.

### Inside the `council` node:

1. **All 5 scholars run independently** — each receives the transcription + context passages and returns:
   - Segments with issues (including `"ambiguous"` severity for semantic near-misses)
   - Proposed readings for ambiguous segments
   - A verdict: `accept` / `needs_editing` / `needs_reocr`
   - A confidence score (0-100)
2. **Consensus is tallied**:
   - Per-segment readings: majority reading wins, applied to `current_best`
   - Verdict: majority vote wins and drives the graph routing
   - Confidence: average across all scholars
3. The node outputs the same state shape as `node_critic` today (`critiques`, `current_score`, `plateau_count`, etc.) so `route_after_critic` works unchanged.

### Arbitrator in `node_reocr` also becomes a council vote:
Same scholar panel, but comparing two transcription versions instead of analyzing one.

---

## Step 1: Config — `ocr_agent/config.py`

Add after existing constants:

```python
# ── Scholar Council ──────────────────────────────────────────────
SCHOLAR_COUNT = 5
SCHOLAR_TEMPERATURES = [0.1, 0.3, 0.5, 0.3, 0.1]

# ── RAG Context Store ───────────────────────────────────────────
# TBD — implementation details to be planned separately
```

## Step 2: State — `ocr_agent/state.py`

Add new fields to `OCRState`:

```python
council_votes: list[dict]            # Vote tallies from each council round
context_passages: list[dict]         # TBD — populated by RAG context retrieval
# TBD — additional RAG-related state fields to be determined
```

## Step 3: New module — `ocr_agent/context_store.py`

TBD — vector store and RAG retrieval implementation to be planned separately.

## Step 4: Scholar agents — `ocr_agent/agents.py`

### 4a. Update `CriticIssue` severity — add `"ambiguous"`
```python
severity: Literal["critical", "minor", "cosmetic", "ambiguous"] = "minor"
```

### 4b. New Pydantic models for scholar output

Each scholar returns the **same shape as `CriticResult`** plus their proposed readings:

```python
class ScholarReading(BaseModel):
    segment_text: str          # The ambiguous segment
    proposed_reading: str      # This scholar's proposed reading
    reasoning: str

class ScholarResult(BaseModel):
    # Same fields as CriticResult (for compatibility)
    overall_confidence: int
    segments: list[CriticSegment]
    verdict: Literal["accept", "needs_editing", "needs_reocr"]
    reasoning: str
    # Scholar-specific additions
    readings: list[ScholarReading]   # Proposed readings for ambiguous segments

class ScholarVote(BaseModel):
    segment_text: str
    winning_reading: str
    vote_counts: dict[str, int]
    total_votes: int

class CouncilResult(BaseModel):
    final_text: str                  # Transcription with winning readings applied
    consensus_verdict: str           # Majority verdict
    consensus_confidence: int        # Average confidence
    votes: list[ScholarVote]         # Per-segment vote tallies
    verdict_tally: dict[str, int]    # e.g. {"accept": 3, "needs_editing": 2}
    merged_critique: CriticResult    # Combined critique for editor compatibility
```

### 4c. Five scholar perspectives

| # | Scholar | Focus | Temp |
|---|---------|-------|------|
| 1 | Grammarian | Grammar, syntax, subject-verb agreement | 0.1 |
| 2 | Contextual Analyst | Semantic coherence, thematic flow | 0.3 |
| 3 | Paleography Expert | Common handwriting confusion patterns (rn/m, cl/d, etc.) | 0.5 |
| 4 | Frequency Analyst | Word frequency, common collocations, idiomatic phrases | 0.3 |
| 5 | Document Specialist | Cross-document/cross-page consistency via RAG context | 0.1 |

Each scholar's system prompt includes the full critic instructions (OCR error patterns, severity levels, etc.) **plus** their specialist lens. They all use the same `ScholarResult` JSON schema.

### 4d. `run_scholar(transcription, perspective, context_passages, temperature)` → `ScholarResult`
Runs one scholar with its specific system prompt and temperature.

### 4e. `run_council(transcription, context_passages, previous_critique)` → `CouncilResult`
- Runs all 5 scholars sequentially
- Tallies segment votes: majority reading wins per ambiguous segment
- Tallies verdict votes: majority verdict wins
- Averages confidence scores
- Builds `merged_critique` (a `CriticResult`) by merging all scholars' issues (union of all flagged segments, highest severity wins for duplicates)
- Builds `final_text` by applying winning readings to the transcription

### 4f. `run_arbitrator_council(versions, context_passages)` → `CouncilResult`
Variant for the reocr path. Same scholar panel but prompt is "compare two versions and vote on which segments to keep". Replaces the single `run_arbitrator()`.

### 4g. Existing `run_critic` and `run_arbitrator` — kept but no longer called from the graph
They remain in the codebase for potential standalone use / testing, but `node_council` and `node_reocr` call the council variants instead.

## Step 5: Graph nodes — `ocr_agent/nodes.py`

### 5a. `node_retrieve_context` (new)
TBD — will retrieve context passages from the RAG store. Implementation details to be planned separately.

### 5b. `node_council` (replaces `node_critic`)
```python
def node_council(state: OCRState) -> dict:
    # Run the full scholar council
    council = run_council(
        transcription=state["current_best"],
        context_passages=state.get("context_passages", []),
        previous_critique=prev_critique,
    )

    # Apply winning readings to current_best
    current_best = council.final_text

    # Store the merged critique in the same format as before
    critiques.append(council.merged_critique.model_dump())

    # Plateau detection (same logic as old node_critic)
    ...

    return {
        "iteration": iteration,
        "critiques": critiques,
        "current_best": current_best,
        "current_score": council.consensus_confidence,
        "council_votes": state["council_votes"] + [council.model_dump()],
        "plateau_count": plateau_count,
        "prev_score": council.consensus_confidence,
        ...
    }
```

Key: `state["critiques"]` still gets populated with a `CriticResult`-shaped dict (the `merged_critique`), so `route_after_critic` and `node_editor` work unchanged.

### 5c. Update `node_reocr`
Replace `run_arbitrator(versions)` with `run_arbitrator_council(versions, context_passages)`. Same state output shape.

### 5d. `node_store_result` (new)
TBD — will store the final transcription in the RAG store for future context retrieval. Implementation details to be planned separately.

### 5e. `node_editor` — unchanged
Still receives the critique (now the council's merged critique) and fixes flagged issues. No changes needed.

## Step 6: Rewire graph — `ocr_agent/graph.py`

```python
def build_ocr_graph():
    builder = StateGraph(OCRState)

    builder.add_node("initial_ocr", node_initial_ocr)
    # builder.add_node("retrieve_context", node_retrieve_context)   # TBD — RAG
    builder.add_node("council", node_council)                     # WAS "critic"
    builder.add_node("editor", node_editor)
    builder.add_node("reocr", node_reocr)                        # internally uses council
    builder.add_node("accept", node_accept)
    builder.add_node("plateau", node_plateau)
    builder.add_node("max_iterations", node_max_iter)
    # builder.add_node("store_result", node_store_result)           # TBD — RAG

    builder.add_edge(START, "initial_ocr")
    # TBD: insert retrieve_context between initial_ocr and council when RAG is implemented
    builder.add_edge("initial_ocr", "council")

    # Same routing as before, from "council" instead of "critic"
    builder.add_conditional_edges("council", route_after_critic, {
        "accept": "accept",
        "plateau": "plateau",
        "max_iterations": "max_iterations",
        "reocr": "reocr",
        "edit": "editor",
    })
    builder.add_edge("editor", "council")
    builder.add_conditional_edges("reocr", route_after_reocr, {
        "council": "council",
        "max_iterations": "max_iterations",
    })

    # Terminal nodes -> END (TBD: insert store_result before END when RAG is implemented)
    builder.add_edge("accept", END)
    builder.add_edge("plateau", END)
    builder.add_edge("max_iterations", END)

    return builder.compile()
```

`route_after_critic` and `route_after_reocr` — logic unchanged, just reads from `state["critiques"]` which the council populates in the same format.

## Step 7: CLI updates — `ocr_agent/transcribe.py`

- Add `--scholar-count` to override `SCHOLAR_COUNT`
- Initialize new state fields: `council_votes: []`, `context_passages: []`
- TBD: additional CLI args for RAG (e.g. `--document-id`, `--page-number`) to be planned separately
- Thread through `transcribe_single()` and batch mode

## Step 8: Dependencies — `pyproject.toml`

TBD — RAG-related dependencies (vector store, embeddings) to be determined after planning.

---

## Files Modified
| File | Change |
|------|--------|
| `ocr_agent/config.py` | Add scholar config constants (RAG config TBD) |
| `ocr_agent/state.py` | Add council fields to OCRState (RAG fields TBD) |
| `ocr_agent/context_store.py` | **NEW** — TBD (vector store + RAG retrieval) |
| `ocr_agent/agents.py` | Add `"ambiguous"` severity, scholar models, `run_council()`, `run_arbitrator_council()` |
| `ocr_agent/nodes.py` | Replace `node_critic` → `node_council`, update `node_reocr` (RAG nodes TBD) |
| `ocr_agent/graph.py` | Rename critic→council (RAG nodes TBD) |
| `ocr_agent/transcribe.py` | New CLI args, new initial state fields (RAG args TBD) |
| `pyproject.toml` | TBD — RAG dependencies |

## Verification

1. Install deps and confirm clean import
2. Run on a known-ambiguous sample — verify:
   - 5 scholars each produce independent analyses
   - Votes are tallied, consensus verdict drives routing
   - Ambiguous segments get majority-voted readings
   - Editor receives the merged critique and fixes flagged issues
3. Run on clean text — verify scholars still run but reach quick consensus (all vote accept)
4. Check trace JSON for council vote details
5. TBD — RAG-specific verification (context retrieval, store_result, cross-document context) to be planned separately
