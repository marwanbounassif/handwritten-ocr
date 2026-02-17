"""
State definition for the LangGraph OCR pipeline.
"""

import time
from datetime import datetime, timezone
from typing import TypedDict


class OCRState(TypedDict):
    """Full state flowing through the LangGraph OCR pipeline."""

    image_path: str
    candidates: list[dict]  # {text, source, ocr_params, score}
    critiques: list[dict]  # CriticResult dicts
    edits: list[dict]  # EditorResult dicts
    current_best: str
    current_score: float
    iteration: int
    max_iterations: int
    status: str  # "running" | "completed" | "max_iterations"
    reason: str  # "accept" | "plateau" | "exhausted" | "max_iterations" | ""
    strategies_used: list[str]
    plateau_count: int
    prev_score: float
    prev_critique: dict | None  # Serialized CriticResult for critic context
    config: dict  # accept_threshold, plateau_patience, strategies, agreement_threshold
    trace_events: list[dict]
    start_time: float


def trace_log(
    state: OCRState,
    *,
    iteration: int,
    agent: str,
    action: str,
    input_summary: str,
    output_summary: str,
    full_input: dict | None = None,
    full_output: dict | None = None,
    metrics: dict | None = None,
    decision: str | None = None,
) -> dict:
    """Create a trace event dict, mirroring Trace.log() format. Returns the event."""
    elapsed = round(time.monotonic() - state["start_time"], 1)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "iteration": iteration,
        "agent": agent,
        "action": action,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "full_input": full_input or {},
        "full_output": full_output or {},
        "metrics": metrics or {},
        "decision": decision,
    }
    # Print live log line
    m, s = divmod(int(elapsed), 60)
    print(f"[{m:02d}:{s:02d}] {output_summary}")
    return event
