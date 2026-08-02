"""Workflow routing helpers."""

from src.graph.state import AgentState
from src.schemas.planner import ExtractionTool


def route_selected_tool(state: AgentState) -> str:
    """Return the workflow branch for the selected extraction tool."""

    selected_tool = state.get("selected_tool", "")
    if selected_tool == ExtractionTool.PDF_PARSER:
        return "pdf_parser"
    if selected_tool == ExtractionTool.OCR:
        return "ocr"
    if selected_tool == ExtractionTool.TEXT_PARSER:
        return "text_parser"
    return "ocr"


def route_reflector(state: AgentState) -> str:
    """Return 'end' if validation passed or max retries reached, else 'retry'."""

    validation_result = state.get("validation_result", {})
    is_valid = bool(validation_result.get("is_valid", False))
    retry_count = state.get("retry_count", 0)

    if state.get("force_finalize"):
        return "end"

    if is_valid or retry_count >= 3:
        return "end"

    if state.get("selected_tool") == "VISION_LLM":
        return "vision_retry"

    return "retry"
