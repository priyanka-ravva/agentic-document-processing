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
            
            # Smart Fallback to Vision LLM for OCR-sourced documents. Two paths trigger it:
            # 1. Retry 1 with no extraction-call error: the model ran cleanly but the
            #    result still failed validation, which usually means the OCR text itself
            #    was garbled - go straight to Vision LLM.
            # 2. Retry 2+ regardless of error: a plain retry on the same OCR text has
            #    already been given one chance. If it's still failing - including a
            #    repeated extraction-call error, which can recur deterministically across
            #    retries rather than being transient - keep retrying the same broken path
            #    is pointless, so escalate to Vision LLM before the retry budget runs out.
            if state.get("selected_tool") == "OCR" and (
                (updated_state["retry_count"] == 1 and not state.get("error"))
                or updated_state["retry_count"] >= 2
            ):
                updated_state["previous_structured_output"] = state.get("structured_output", {})
                updated_state["previous_document_type"] = state.get("document_type", "")
                updated_state["selected_tool"] = "VISION_LLM"
                message = "Reflection completed. OCR-based extraction was unsuccessful; falling back to Multimodal Vision LLM."
            else:
                message = f"Reflection completed. Extraction has gaps; initiating retry {updated_state['retry_count']}."

        return add_log(
            updated_state,
            agent=self.name,
            message=message,
            is_valid=is_valid,
            quality_score=quality_score,
        )
