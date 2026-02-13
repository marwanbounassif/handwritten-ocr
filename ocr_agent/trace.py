"""
Trace logging and observability for the agentic OCR pipeline.
Every action gets logged with full input/output for debugging.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class Trace:
    """Collects timestamped events throughout a pipeline run."""

    def __init__(self):
        self.events: list[dict] = []
        self._start_time = time.monotonic()

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    def log(
        self,
        iteration: int,
        agent: str,
        action: str,
        input_summary: str,
        output_summary: str,
        full_input: dict | None = None,
        full_output: dict | None = None,
        metrics: dict | None = None,
        decision: str | None = None,
    ):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(self._elapsed(), 1),
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
        self.events.append(event)
        # Print a live log line so the user can follow along
        elapsed = self._format_elapsed(event["elapsed_seconds"])
        print(f"[{elapsed}] {output_summary}")

    def _format_elapsed(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def save_json(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)

    def save_summary(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for e in self.events:
            elapsed = self._format_elapsed(e["elapsed_seconds"])
            lines.append(f"[{elapsed}] {e['output_summary']}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def print_summary(self):
        for e in self.events:
            elapsed = self._format_elapsed(e["elapsed_seconds"])
            print(f"[{elapsed}] {e['output_summary']}")
