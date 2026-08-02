"""Planner agent."""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.config import get_llm, get_model_names
from src.graph.state import AgentState, add_log
from src.prompts.prompt_factory import PromptFactory
from src.schemas.planner import ExtractionTool, PlannerDecision


class PlannerAgent(BaseAgent):
    """Decides which extraction strategy should be used."""

    name = "planner"

    def invoke(self, state: AgentState) -> AgentState:
        """Select an extraction tool based on document metadata."""

        prompt = PromptFactory.get_prompt(agent=self.name, context=state)

        if self.llm:
            try:
                return self._plan_with_llm(
                    state=state,
                    system_prompt=prompt.system_prompt,
                    llm=self.llm,
                    model_name="injected",
                )
            except Exception as exc:
                return self._plan_with_rules(
                    state,
                    fallback_reason=f"Injected planner LLM failed: {exc}",
                )

        errors: list[str] = []
        for model_name in get_model_names():
            try:
                llm = get_llm(model_name=model_name)
                return self._plan_with_llm(
                    state=state,
                    system_prompt=prompt.system_prompt,
                    llm=llm,
                    model_name=model_name,
                )
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")

        return self._plan_with_rules(
            state,
            fallback_reason="Planner LLM failed for all configured models.",
            errors=errors,
        )

    def _plan_with_llm(
        self,
        state: AgentState,
        system_prompt: str,
        llm,
        model_name: str,
    ) -> AgentState:
        """Ask an LLM for a structured planner decision."""

        metadata = state.get("document_metadata", {})
        structured_llm = llm.with_structured_output(PlannerDecision)
        decision = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        "Choose the best first extraction tool for this document metadata. "
                        "Return only the structured planner decision.\n\n"
                        f"{json.dumps(metadata, indent=2, default=str)}"
                    )
                ),
            ]
        )

        return self._apply_decision(
            state=state,
            decision=decision,
            source="llm",
            model=model_name,
            message="Planner selected extraction tool with LLM reasoning summary.",
        )

    def _plan_with_rules(
        self,
        state: AgentState,
        fallback_reason: str | None = None,
        errors: list[str] | None = None,
    ) -> AgentState:
        """Select an extraction tool using deterministic metadata rules."""

        metadata = state.get("document_metadata", {})
        file_extension = metadata.get("file_extension", "")
        text_length = int(metadata.get("text_length", 0))
        has_embedded_text = bool(metadata.get("has_embedded_text", False))
        supported_text_extensions = {".txt", ".json", ".csv", ".xlsx", ".docx"}

        if file_extension in supported_text_extensions:
            decision = PlannerDecision(
                selected_tool=ExtractionTool.TEXT_PARSER,
                reasoning=(
                    "The input file is a plain text or structured-text document, so TEXT_PARSER is the best first tool."
                ),
                confidence=0.9,
            )
        elif has_embedded_text and text_length >= 100:
            decision = PlannerDecision(
                selected_tool=ExtractionTool.PDF_PARSER,
                reasoning="Embedded text was found with enough content, so PDF_PARSER is the best first tool.",
                confidence=0.9,
            )
        else:
            decision = PlannerDecision(
                selected_tool=ExtractionTool.OCR,
                reasoning="Little or no embedded text was found, so OCR is the best first tool.",
                confidence=0.85,
            )

        return self._apply_decision(
            state=state,
            decision=decision,
            source="rules",
            message="Planner selected extraction tool with deterministic fallback rules.",
            fallback_reason=fallback_reason,
            errors=errors,
        )

    def _apply_decision(
        self,
        state: AgentState,
        decision: PlannerDecision,
        source: str,
        message: str,
        model: str | None = None,
        fallback_reason: str | None = None,
        errors: list[str] | None = None,
    ) -> AgentState:
        """Persist a planner decision into workflow state and logs."""

        updated_state = state.copy()
        updated_state["selected_tool"] = decision.selected_tool.value
        updated_state["planner_reasoning"] = decision.reasoning

        log_metadata = {
            "selected_tool": decision.selected_tool.value,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "source": source,
        }
        if model:
            log_metadata["model"] = model
        if fallback_reason:
            log_metadata["fallback_reason"] = fallback_reason
        if errors:
            log_metadata["errors"] = errors

        return add_log(updated_state, agent=self.name, message=message, **log_metadata)
