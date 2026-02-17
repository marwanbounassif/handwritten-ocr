"""
LangGraph node functions for the agentic OCR pipeline.
Each node receives OCRState and returns a partial state update dict.
"""

from ocr_agent.agents import CriticResult, run_arbitrator, run_critic, run_editor
from ocr_agent.state import OCRState, trace_log
from ocr_agent.tools import (
    compare_versions,
    merge_versions,
    preprocess_image,
    run_ocr,
    unload_ocr_model,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _strategy_label(strategy: str | list[str]) -> str:
    """Human-readable label for a strategy (single string or pipeline list)."""
    if isinstance(strategy, list):
        return "+".join(strategy)
    return strategy


def _do_ocr_pass(
    state: OCRState,
    strategy: str | list[str],
    candidates: list[dict],
    strategies_used: list[str],
    trace_events: list[dict],
    iteration: int = 0,
) -> None:
    """Run a single OCR pass. Mutates candidates, strategies_used, trace_events in place."""
    label = _strategy_label(strategy)
    if label in strategies_used:
        return
    strategies_used.append(label)

    processed_path = preprocess_image(state["image_path"], strategy)
    trace_events.append(trace_log(
        state,
        iteration=iteration,
        agent="reader",
        action="preprocess",
        input_summary=f"Image: {state['image_path']}",
        output_summary=f"Preprocessed with '{label}'",
        metrics={"strategy": label},
    ))

    text = run_ocr(processed_path)
    candidate = {
        "text": text,
        "source": f"ocr_{label}",
        "ocr_params": {"strategy": label},
        "score": None,
    }
    candidates.append(candidate)

    trace_events.append(trace_log(
        state,
        iteration=iteration,
        agent="reader",
        action="ocr",
        input_summary=f"Preprocessed image ({label})",
        output_summary=f"OCR pass ({label}) → {len(text)} chars",
        full_output={"text_preview": text[:200]},
        metrics={"chars": len(text), "strategy": label},
    ))


# ── Node functions ───────────────────────────────────────────────────


def node_initial_ocr(state: OCRState) -> dict:
    """Phase 1: multi-strategy OCR reads, agreement check, merge candidates."""
    print("\n=== PHASE 1: Initial OCR Reads ===")

    strategy_list = state["config"]["strategies"]
    candidates = list(state["candidates"])
    strategies_used = list(state["strategies_used"])
    trace_events = []

    # First OCR pass
    _do_ocr_pass(state, strategy_list[0] if strategy_list else "original",
                 candidates, strategies_used, trace_events)

    # Second OCR pass
    if len(strategy_list) > 1:
        _do_ocr_pass(state, strategy_list[1], candidates, strategies_used, trace_events)

    # Check agreement between first two candidates
    if len(candidates) >= 2:
        cmp = compare_versions(candidates[0]["text"], candidates[1]["text"])
        agreement = cmp["agreement_rate"]
        trace_events.append(trace_log(
            state,
            iteration=0,
            agent="orchestrator",
            action="compare",
            input_summary="Comparing candidate 1 vs 2",
            output_summary=f"Versions agree {agreement}%",
            full_output=cmp,
            metrics={"agreement_rate": agreement},
            decision="tiebreaker" if agreement < state["config"]["agreement_threshold"] else "merge",
        ))

        if agreement < state["config"]["agreement_threshold"] and len(strategy_list) > 2:
            _do_ocr_pass(state, strategy_list[2], candidates, strategies_used, trace_events)

    # Merge initial candidates
    candidate_texts = [c["text"] for c in candidates]
    current_best = merge_versions(candidate_texts)
    trace_events.append(trace_log(
        state,
        iteration=0,
        agent="orchestrator",
        action="merge",
        input_summary=f"Merging {len(candidate_texts)} candidates",
        output_summary=f"Merged → {len(current_best)} chars",
        metrics={"merged_chars": len(current_best)},
    ))

    # Unload OCR model to free memory for LLM agents
    print("\n--- Unloading OCR model to free memory for LLM agents ---")
    unload_ocr_model()

    return {
        "candidates": candidates,
        "current_best": current_best,
        "strategies_used": strategies_used,
        "trace_events": state["trace_events"] + trace_events,
    }


def node_critic(state: OCRState) -> dict:
    """Run critic agent, update score and plateau tracking."""
    iteration = state["iteration"] + 1
    if iteration == 1:
        print("\n=== PHASE 2: Critique-Edit Loop ===")
    print(f"\n--- Iteration {iteration} ---")

    # Reconstruct previous critique if available
    prev_critique = None
    if state["prev_critique"]:
        try:
            prev_critique = CriticResult.model_validate(state["prev_critique"])
        except Exception:
            pass

    critique = run_critic(state["current_best"], previous_critique=prev_critique)
    critiques = list(state["critiques"])
    critiques.append(critique.model_dump())

    confidence = critique.overall_confidence
    verdict = critique.verdict
    n_issues = sum(len(seg.issues) for seg in critique.segments)
    n_critical = sum(
        1 for seg in critique.segments for issue in seg.issues if issue.severity == "critical"
    )
    n_minor = sum(
        1 for seg in critique.segments for issue in seg.issues if issue.severity == "minor"
    )
    n_cosmetic = n_issues - n_critical - n_minor

    trace_events = [trace_log(
        state,
        iteration=iteration,
        agent="critic",
        action="critique",
        input_summary=f"Transcription ({len(state['current_best'])} chars)",
        output_summary=(
            f"Critic: confidence {confidence}, verdict={verdict} "
            f"({n_issues} issues: {n_critical} critical, {n_minor} minor, {n_cosmetic} cosmetic)"
        ),
        full_input={"transcription": state["current_best"]},
        full_output=critique.model_dump(),
        metrics={
            "confidence": confidence,
            "n_issues": n_issues,
            "n_critical": n_critical,
            "n_minor": n_minor,
            "n_cosmetic": n_cosmetic,
        },
        decision=verdict,
    )]

    # Plateau detection
    plateau_count = state["plateau_count"]
    if confidence <= state["prev_score"]:
        plateau_count += 1
    else:
        plateau_count = 0

    return {
        "iteration": iteration,
        "critiques": critiques,
        "current_score": confidence,
        "plateau_count": plateau_count,
        "prev_score": confidence,
        "trace_events": state["trace_events"] + trace_events,
    }


def node_editor(state: OCRState) -> dict:
    """Run editor agent to fix issues identified by critic."""
    latest_critique_dict = state["critiques"][-1]
    critique = CriticResult.model_validate(latest_critique_dict)
    n_issues = sum(len(seg.issues) for seg in critique.segments)

    edit_result = run_editor(state["current_best"], critique)
    edits = list(state["edits"])
    edits.append(edit_result.model_dump())

    n_changes = len(edit_result.changes)
    n_unresolved = len(edit_result.unresolved)

    trace_events = [trace_log(
        state,
        iteration=state["iteration"],
        agent="editor",
        action="edit",
        input_summary=f"Transcription + {n_issues} critic issues",
        output_summary=f"Editor: fixed {n_changes} issues, {n_unresolved} unresolved",
        full_input={"transcription": state["current_best"], "critique": latest_critique_dict},
        full_output=edit_result.model_dump(),
        metrics={"changes_made": n_changes, "unresolved": n_unresolved},
    )]

    return {
        "current_best": edit_result.corrected_text,
        "edits": edits,
        "prev_critique": latest_critique_dict,
        "trace_events": state["trace_events"] + trace_events,
    }


def node_reocr(state: OCRState) -> dict:
    """Re-OCR with the next unused preprocessing strategy, then arbitrate."""
    strategy_list = state["config"]["strategies"]
    strategies_used = list(state["strategies_used"])
    candidates = list(state["candidates"])
    trace_events = []

    # Find next unused strategy
    next_strategy = None
    for s in strategy_list:
        if _strategy_label(s) not in strategies_used:
            next_strategy = s
            break

    if next_strategy is None:
        # All strategies exhausted
        return {
            "reason": "exhausted",
            "trace_events": state["trace_events"],
        }

    print(f"\n--- Re-OCR with strategy: {_strategy_label(next_strategy)} ---")
    _do_ocr_pass(state, next_strategy, candidates, strategies_used, trace_events,
                 iteration=state["iteration"])

    # Unload OCR model again
    unload_ocr_model()

    # Use arbitrator to merge new candidate with current best
    new_candidate = candidates[-1]
    versions = [
        {"text": state["current_best"], "source": "current_best", "score": state["current_score"]},
        {"text": new_candidate["text"], "source": new_candidate["source"]},
    ]

    arb_result = run_arbitrator(versions)

    trace_events.append(trace_log(
        state,
        iteration=state["iteration"],
        agent="arbitrator",
        action="arbitrate",
        input_summary=f"Current best vs {new_candidate['source']}",
        output_summary=(
            f"Arbitrator: merged with confidence {arb_result.confidence}, "
            f"{len(arb_result.uncertain_segments)} uncertain segments"
        ),
        full_output=arb_result.model_dump(),
        metrics={
            "confidence": arb_result.confidence,
            "n_decisions": len(arb_result.decisions),
            "n_uncertain": len(arb_result.uncertain_segments),
        },
    ))

    latest_critique_dict = state["critiques"][-1] if state["critiques"] else None

    return {
        "current_best": arb_result.final_text,
        "candidates": candidates,
        "strategies_used": strategies_used,
        "prev_critique": latest_critique_dict,
        "trace_events": state["trace_events"] + trace_events,
    }


# ── Terminal nodes ───────────────────────────────────────────────────


def node_accept(state: OCRState) -> dict:
    """Terminal: transcription accepted."""
    trace_events = [trace_log(
        state,
        iteration=state["iteration"],
        agent="orchestrator",
        action="accept",
        input_summary=f"Confidence {state['current_score']} >= {state['config']['accept_threshold']}",
        output_summary=(
            f"DONE — {state['iteration']} iterations, final confidence {state['current_score']}"
        ),
        decision="accept",
    )]
    return {
        "status": "completed",
        "reason": "accept",
        "trace_events": state["trace_events"] + trace_events,
    }


def node_plateau(state: OCRState) -> dict:
    """Terminal: no improvement for N iterations."""
    trace_events = [trace_log(
        state,
        iteration=state["iteration"],
        agent="orchestrator",
        action="plateau",
        input_summary=f"No improvement for {state['config']['plateau_patience']} iterations",
        output_summary=(
            f"DONE (plateau) — {state['iteration']} iterations, "
            f"final confidence {state['current_score']}"
        ),
        decision="plateau_stop",
    )]
    return {
        "status": "completed",
        "reason": "plateau",
        "trace_events": state["trace_events"] + trace_events,
    }


def node_max_iter(state: OCRState) -> dict:
    """Terminal: max iterations or strategies exhausted."""
    reason = state.get("reason", "")
    if reason == "exhausted":
        action = "strategies_exhausted"
        decision = "exhausted_stop"
        summary = (
            f"DONE (strategies exhausted) — {state['iteration']} iterations, "
            f"final confidence {state['current_score']}"
        )
        input_summary = "All preprocessing strategies tried"
    else:
        action = "max_iterations"
        decision = "max_iterations_stop"
        summary = (
            f"DONE (max iterations) — {state['iteration']} iterations, "
            f"final confidence {state['current_score']}"
        )
        input_summary = f"Reached {state['max_iterations']} iterations"

    trace_events = [trace_log(
        state,
        iteration=state["iteration"],
        agent="orchestrator",
        action=action,
        input_summary=input_summary,
        output_summary=summary,
        decision=decision,
    )]
    return {
        "status": "max_iterations" if reason != "exhausted" else "completed",
        "reason": reason or "max_iterations",
        "trace_events": state["trace_events"] + trace_events,
    }
