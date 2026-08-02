"""Utilities for persisting workflow traces."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.graph.state import AgentState


def save_run_trace(state: AgentState, output_dir: str = "logs/runs") -> Path:
    """Save the final workflow state as a JSON trace file."""

    runs_dir = Path(output_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    source_name = Path(state["file_path"]).stem or "document"
    trace_path = runs_dir / f"{timestamp}_{source_name}_trace.json"

    payload: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "file_path": state["file_path"],
        "document_metadata": state.get("document_metadata", {}),
        "selected_tool": state.get("selected_tool", ""),
        "planner_reasoning": state.get("planner_reasoning", ""),
        "structured_output": state.get("structured_output", {}),
        "validation_result": state.get("validation_result", {}),
        "retry_count": state.get("retry_count", 0),
        "logs": state.get("logs", []),
        "error": state.get("error"),
    }

    trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return trace_path
