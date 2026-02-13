"""
Rule-based orchestrator for the agentic OCR pipeline.
A deterministic state machine — NOT an LLM. Manages the workflow loop.
"""

from dataclasses import dataclass, field

from ocr_agent import config
from ocr_agent.agents import run_arbitrator, run_critic, run_editor
from ocr_agent.tools import (
    compare_versions,
    merge_versions,
    preprocess_image,
    run_ocr,
    unload_ocr_model,
)
from ocr_agent.trace import Trace


@dataclass
class TranscriptionState:
    """Holds all state across the pipeline run."""

    image_path: str
    candidates: list[dict] = field(default_factory=list)  # {text, source, ocr_params, score}
    critiques: list[dict] = field(default_factory=list)
    edits: list[dict] = field(default_factory=list)
    current_best: str = ""
    current_score: float = 0.0
    iteration: int = 0
    status: str = "running"  # "running" | "completed" | "max_iterations"
    _strategies_used: list[str] = field(default_factory=list)
    _plateau_count: int = 0
    _prev_score: float = 0.0


def run_pipeline(
    image_path: str,
    max_iterations: int | None = None,
    accept_threshold: int | None = None,
    plateau_patience: int | None = None,
    strategies: list[str] | None = None,
) -> tuple[TranscriptionState, Trace]:
    """
    Run the full agentic OCR pipeline on a single image.
    Returns (final_state, trace).
    """
    max_iter = max_iterations or config.MAX_ITERATIONS
    threshold = accept_threshold or config.ACCEPT_THRESHOLD
    patience = plateau_patience or config.PLATEAU_PATIENCE
    strategy_list = strategies or list(config.PREPROCESSING_STRATEGIES)

    state = TranscriptionState(image_path=image_path)
    trace = Trace()

    # ── PHASE 1: INITIAL READS ────────────────────────────────────
    print("\n=== PHASE 1: Initial OCR Reads ===")
    _do_ocr_pass(state, trace, strategy_list[0] if strategy_list else "original", strategy_list)
    if len(strategy_list) > 1:
        _do_ocr_pass(state, trace, strategy_list[1], strategy_list)

    # Check agreement between first two candidates
    if len(state.candidates) >= 2:
        cmp = compare_versions(state.candidates[0]["text"], state.candidates[1]["text"])
        agreement = cmp["agreement_rate"]
        trace.log(
            iteration=0,
            agent="orchestrator",
            action="compare",
            input_summary=f"Comparing candidate 1 vs 2",
            output_summary=f"Versions agree {agreement}%",
            full_output=cmp,
            metrics={"agreement_rate": agreement},
            decision="tiebreaker" if agreement < config.AGREEMENT_THRESHOLD else "merge",
        )

        if agreement < config.AGREEMENT_THRESHOLD and len(strategy_list) > 2:
            _do_ocr_pass(state, trace, strategy_list[2], strategy_list)

    # Merge initial candidates
    candidate_texts = [c["text"] for c in state.candidates]
    state.current_best = merge_versions(candidate_texts)
    trace.log(
        iteration=0,
        agent="orchestrator",
        action="merge",
        input_summary=f"Merging {len(candidate_texts)} candidates",
        output_summary=f"Merged → {len(state.current_best)} chars",
        metrics={"merged_chars": len(state.current_best)},
    )

    # Unload OCR model to free memory for LLM agents
    print("\n--- Unloading OCR model to free memory for LLM agents ---")
    unload_ocr_model()

    # ── PHASE 2: CRITIQUE-EDIT LOOP ──────────────────────────────
    print("\n=== PHASE 2: Critique-Edit Loop ===")
    prev_critique = None

    while state.status == "running" and state.iteration < max_iter:
        state.iteration += 1
        print(f"\n--- Iteration {state.iteration} ---")

        # CRITIQUE
        critique = run_critic(state.current_best, previous_critique=prev_critique)
        state.critiques.append(critique)

        confidence = critique.get("overall_confidence", 0)
        verdict = critique.get("verdict", "needs_editing")
        n_issues = sum(len(seg.get("issues", [])) for seg in critique.get("segments", []))
        n_critical = sum(
            1
            for seg in critique.get("segments", [])
            for issue in seg.get("issues", [])
            if issue.get("severity") == "critical"
        )
        n_minor = sum(
            1
            for seg in critique.get("segments", [])
            for issue in seg.get("issues", [])
            if issue.get("severity") == "minor"
        )
        n_cosmetic = n_issues - n_critical - n_minor

        trace.log(
            iteration=state.iteration,
            agent="critic",
            action="critique",
            input_summary=f"Transcription ({len(state.current_best)} chars)",
            output_summary=(
                f"Critic: confidence {confidence}, verdict={verdict} "
                f"({n_issues} issues: {n_critical} critical, {n_minor} minor, {n_cosmetic} cosmetic)"
            ),
            full_input={"transcription": state.current_best},
            full_output=critique,
            metrics={
                "confidence": confidence,
                "n_issues": n_issues,
                "n_critical": n_critical,
                "n_minor": n_minor,
                "n_cosmetic": n_cosmetic,
            },
            decision=verdict,
        )

        state.current_score = confidence

        # Check acceptance
        if verdict == "accept" or confidence >= threshold:
            state.status = "completed"
            trace.log(
                iteration=state.iteration,
                agent="orchestrator",
                action="accept",
                input_summary=f"Confidence {confidence} >= {threshold}",
                output_summary=(
                    f"DONE — {state.iteration} iterations, final confidence {confidence}"
                ),
                decision="accept",
            )
            break

        # Check plateau
        if confidence <= state._prev_score:
            state._plateau_count += 1
        else:
            state._plateau_count = 0
        state._prev_score = confidence

        if state._plateau_count >= patience:
            state.status = "completed"
            trace.log(
                iteration=state.iteration,
                agent="orchestrator",
                action="plateau",
                input_summary=f"No improvement for {patience} iterations",
                output_summary=(
                    f"DONE (plateau) — {state.iteration} iterations, "
                    f"final confidence {confidence}"
                ),
                decision="plateau_stop",
            )
            break

        # NEEDS_REOCR path
        if verdict == "needs_reocr":
            reocr_done = _handle_reocr(state, trace, strategy_list, threshold)
            if not reocr_done:
                # All strategies exhausted
                state.status = "completed"
                trace.log(
                    iteration=state.iteration,
                    agent="orchestrator",
                    action="strategies_exhausted",
                    input_summary="All preprocessing strategies tried",
                    output_summary=(
                        f"DONE (strategies exhausted) — {state.iteration} iterations, "
                        f"final confidence {confidence}"
                    ),
                    decision="exhausted_stop",
                )
                break
            prev_critique = critique
            continue

        # NEEDS_EDITING path
        edit_result = run_editor(state.current_best, critique)
        state.edits.append(edit_result)

        n_changes = len(edit_result.get("changes", []))
        n_unresolved = len(edit_result.get("unresolved", []))

        trace.log(
            iteration=state.iteration,
            agent="editor",
            action="edit",
            input_summary=f"Transcription + {n_issues} critic issues",
            output_summary=f"Editor: fixed {n_changes} issues, {n_unresolved} unresolved",
            full_input={"transcription": state.current_best, "critique": critique},
            full_output=edit_result,
            metrics={"changes_made": n_changes, "unresolved": n_unresolved},
        )

        state.current_best = edit_result.get("corrected_text", state.current_best)
        prev_critique = critique

    # Max iterations reached
    if state.status == "running":
        state.status = "max_iterations"
        trace.log(
            iteration=state.iteration,
            agent="orchestrator",
            action="max_iterations",
            input_summary=f"Reached {max_iter} iterations",
            output_summary=(
                f"DONE (max iterations) — {state.iteration} iterations, "
                f"final confidence {state.current_score}"
            ),
            decision="max_iterations_stop",
        )

    return state, trace


def _strategy_label(strategy: str | list[str]) -> str:
    """Human-readable label for a strategy (single string or pipeline list)."""
    if isinstance(strategy, list):
        return "+".join(strategy)
    return strategy


def _do_ocr_pass(
    state: TranscriptionState,
    trace: Trace,
    strategy: str | list[str],
    strategy_list: list,
):
    """Run a single OCR pass with the given preprocessing strategy (or pipeline)."""
    label = _strategy_label(strategy)
    if label in state._strategies_used:
        return
    state._strategies_used.append(label)

    # Preprocess (handles both single strings and lists)
    processed_path = preprocess_image(state.image_path, strategy)
    trace.log(
        iteration=0,
        agent="reader",
        action="preprocess",
        input_summary=f"Image: {state.image_path}",
        output_summary=f"Preprocessed with '{label}'",
        metrics={"strategy": label},
    )

    # OCR
    text = run_ocr(processed_path)
    candidate = {
        "text": text,
        "source": f"ocr_{label}",
        "ocr_params": {"strategy": label},
        "score": None,
    }
    state.candidates.append(candidate)

    trace.log(
        iteration=0,
        agent="reader",
        action="ocr",
        input_summary=f"Preprocessed image ({label})",
        output_summary=f"OCR pass ({label}) → {len(text)} chars",
        full_output={"text_preview": text[:200]},
        metrics={"chars": len(text), "strategy": label},
    )


def _handle_reocr(
    state: TranscriptionState,
    trace: Trace,
    strategy_list: list[str],
    threshold: int,
) -> bool:
    """
    Handle a needs_reocr verdict by trying the next unused preprocessing strategy.
    Returns True if a re-OCR was performed, False if all strategies are exhausted.
    """
    # Find next unused strategy
    next_strategy = None
    for s in strategy_list:
        if _strategy_label(s) not in state._strategies_used:
            next_strategy = s
            break

    if next_strategy is None:
        return False

    # Need to reload OCR model for re-OCR
    print(f"\n--- Re-OCR with strategy: {_strategy_label(next_strategy)} ---")
    _do_ocr_pass(state, trace, next_strategy, strategy_list)

    # Unload OCR model again
    unload_ocr_model()

    # Use arbitrator to merge new candidate with current best
    new_candidate = state.candidates[-1]
    versions = [
        {"text": state.current_best, "source": "current_best", "score": state.current_score},
        {"text": new_candidate["text"], "source": new_candidate["source"]},
    ]

    arb_result = run_arbitrator(versions)

    trace.log(
        iteration=state.iteration,
        agent="arbitrator",
        action="arbitrate",
        input_summary=f"Current best vs {new_candidate['source']}",
        output_summary=(
            f"Arbitrator: merged with confidence {arb_result.get('confidence', '?')}, "
            f"{len(arb_result.get('uncertain_segments', []))} uncertain segments"
        ),
        full_output=arb_result,
        metrics={
            "confidence": arb_result.get("confidence", 0),
            "n_decisions": len(arb_result.get("decisions", [])),
            "n_uncertain": len(arb_result.get("uncertain_segments", [])),
        },
    )

    state.current_best = arb_result.get("final_text", state.current_best)
    return True
