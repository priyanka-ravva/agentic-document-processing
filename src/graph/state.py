"""Shared LangGraph state definitions."""

from typing import Any, NotRequired, TypedDict


class AgentState(TypedDict):
    """State passed between LangGraph workflow nodes."""

    file_path: str
    document_metadata: dict[str, Any]
    planner_reasoning: str
    selected_tool: str
    extracted_text: str
    document_type: str
    structured_output: dict[str, Any]
    validation_result: dict[str, Any]
    previous_structured_output: NotRequired[dict[str, Any]]
    previous_document_type: NotRequired[str]
    force_finalize: NotRequired[bool]
    retry_count: int
    logs: list[dict[str, Any]]
    error: NotRequired[str]


def create_initial_state(file_path: str) -> AgentState:
    """Create an empty workflow state for a document path."""

    return {
        "file_path": file_path,
        "document_metadata": {},
        "planner_reasoning": "",
        "selected_tool": "",
        "extracted_text": "",
        "document_type": "",
        "structured_output": {},
        "validation_result": {},
        "retry_count": 0,
        "logs": [],
    }


def add_log(state: AgentState, agent: str, message: str, **metadata: Any) -> AgentState:
    """Append a structured log event to the state."""

    updated_state = state.copy()
    updated_logs = list(updated_state.get("logs", []))
    updated_logs.append(
        {
            "agent": agent,
            "message": message,
            "metadata": metadata,
        }
    )
    updated_state["logs"] = updated_logs
    return updated_state
