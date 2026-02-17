"""
LangGraph definition for the agentic OCR pipeline.
"""

from langgraph.graph import END, START, StateGraph

from ocr_agent.nodes import (
    node_accept,
    node_critic,
    node_editor,
    node_initial_ocr,
    node_max_iter,
    node_plateau,
    node_reocr,
)
from ocr_agent.state import OCRState


# ── Routing functions ────────────────────────────────────────────────


def route_after_critic(state: OCRState) -> str:
    """Decide next step after critic evaluates the transcription."""
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
    """After re-OCR, check if strategies are exhausted."""
    if state.get("reason") == "exhausted":
        return "max_iterations"
    return "critic"


# ── Graph builder ────────────────────────────────────────────────────


def build_ocr_graph():
    """Build and compile the OCR pipeline graph."""
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
        "accept": "accept",
        "plateau": "plateau",
        "max_iterations": "max_iterations",
        "reocr": "reocr",
        "edit": "editor",
    })
    builder.add_edge("editor", "critic")
    builder.add_conditional_edges("reocr", route_after_reocr, {
        "critic": "critic",
        "max_iterations": "max_iterations",
    })
    builder.add_edge("accept", END)
    builder.add_edge("plateau", END)
    builder.add_edge("max_iterations", END)

    return builder.compile()
