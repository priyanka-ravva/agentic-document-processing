"""Reflection agent."""

from src.agents.base_agent import BaseAgent
from src.graph.state import AgentState, add_log


class ReflectionAgent(BaseAgent):
    """Decides whether the workflow should retry or finalize."""

    name = "reflector"

    def invoke(self, state: AgentState) -> AgentState:
        """Decides whether the workflow should retry or finalize."""

        validation_result = state.get("validation_result", {})
        is_valid = bool(validation_result.get("is_valid", False))
        quality_score = validation_result.get("quality_score", 0.0)
        updated_state = state.copy()

        if is_valid:
            message = "Reflection completed. Extraction is acceptable for final response."
        else:
            updated_state["retry_count"] = state.get("retry_count", 0) + 1
            
            # Smart Fallback: If OCR extraction failed validation, it's likely garbled text.
            # We switch the selected tool to VISION_LLM to read directly from the image.
            if state.get("selected_tool") == "OCR" and updated_state["retry_count"] == 1:
                updated_state["previous_structured_output"] = state.get("structured_output", {})
                updated_state["previous_document_type"] = state.get("document_type", "")
                updated_state["selected_tool"] = "VISION_LLM"
                message = "Reflection completed. OCR extraction was garbled; falling back to Multimodal Vision LLM."
            else:
                message = f"Reflection completed. Extraction has gaps; initiating retry {updated_state['retry_count']}."

        return add_log(
            updated_state,
            agent=self.name,
            message=message,
            is_valid=is_valid,
            quality_score=quality_score,
        )
