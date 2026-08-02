"""Classifier agent."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.config import get_llm, get_model_names
from src.graph.state import AgentState, add_log
from src.prompts.prompt_factory import PromptFactory
from src.schemas.extraction import DocumentType


class ClassificationResult(BaseModel):
    """Output of the classifier agent."""

    document_type: DocumentType = Field(description="Best classification for the document.")


class ClassifierAgent(BaseAgent):
    """Classifies document text into a specific document type."""

    name = "classifier"

    def invoke(self, state: AgentState) -> AgentState:
        """Classify the document text."""

        extracted_text = state.get("extracted_text", "").strip()
        updated_state = state.copy()

        if not extracted_text:
            updated_state["document_type"] = DocumentType.UNKNOWN.value
            return add_log(
                updated_state,
                agent=self.name,
                message="Classification skipped because extracted_text was empty.",
                document_type=DocumentType.UNKNOWN.value,
            )

        prompt = PromptFactory.get_prompt(agent=self.name, context=state)

        if self.llm:
            return self._classify_with_llm(state, updated_state, extracted_text, prompt.system_prompt, self.llm, "injected")

        errors: list[str] = []
        for model_name in get_model_names():
            try:
                llm = get_llm(model_name=model_name)
                return self._classify_with_llm(
                    state,
                    updated_state,
                    extracted_text,
                    prompt.system_prompt,
                    llm,
                    model_name,
                )
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")

        error_message = " | ".join(errors)
        updated_state["document_type"] = DocumentType.UNKNOWN.value
        updated_state["error"] = error_message
        return add_log(
            updated_state,
            agent=self.name,
            message="Classification failed for all configured models; fallback to unknown.",
            errors=errors,
        )

    def _classify_with_llm(
        self,
        state: AgentState,
        updated_state: AgentState,
        extracted_text: str,
        system_prompt: str,
        llm,
        model_name: str,
    ) -> AgentState:
        """Run classification with one configured model."""

        try:
            structured_llm = llm.with_structured_output(ClassificationResult)
            response = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Classify this document text:\n\n{extracted_text[:4000]}"),
                ]
            )

            updated_state["document_type"] = response.document_type.value
            return add_log(
                updated_state,
                agent=self.name,
                message="Classification completed.",
                document_type=updated_state["document_type"],
                model=model_name,
            )
        except Exception as exc:
            raise RuntimeError(f"Classification failed with {model_name}: {exc}") from exc
