"""Tests for the hybrid planner agent."""

import sys
import types

langchain_groq = types.ModuleType("langchain_groq")
langchain_groq.ChatGroq = object
sys.modules.setdefault("langchain_groq", langchain_groq)

from src.agents.planner import PlannerAgent
from src.graph.state import create_initial_state
from src.schemas.planner import ExtractionTool, PlannerDecision


class FakeStructuredPlanner:
    """Fake structured LLM wrapper returning a planner decision."""

    def __init__(self, decision: PlannerDecision | None = None, error: Exception | None = None) -> None:
        self.decision = decision
        self.error = error

    def invoke(self, messages):
        """Return a decision or raise an injected error."""

        if self.error:
            raise self.error
        return self.decision


class FakePlannerLlm:
    """Fake LLM that supports LangChain structured output."""

    def __init__(self, structured_planner: FakeStructuredPlanner) -> None:
        self.structured_planner = structured_planner

    def with_structured_output(self, schema):
        """Return the fake structured planner."""

        return self.structured_planner


def test_planner_uses_injected_llm_decision() -> None:
    """Planner should use an LLM decision when structured planning succeeds."""

    state = create_initial_state("sample_docs/invoice_simple.pdf")
    state["document_metadata"] = {
        "file_extension": ".pdf",
        "text_length": 0,
        "has_embedded_text": False,
    }
    decision = PlannerDecision(
        selected_tool=ExtractionTool.PDF_PARSER,
        reasoning="The LLM selected PDF parsing for this test.",
        confidence=0.77,
    )

    result = PlannerAgent(
        llm=FakePlannerLlm(FakeStructuredPlanner(decision=decision))
    ).invoke(state)

    assert result["selected_tool"] == "PDF_PARSER"
    assert result["planner_reasoning"] == "The LLM selected PDF parsing for this test."
    assert result["logs"][-1]["metadata"]["source"] == "llm"
    assert result["logs"][-1]["metadata"]["model"] == "injected"


def test_planner_falls_back_to_rules_when_injected_llm_fails() -> None:
    """Planner should preserve deterministic behavior if LLM planning fails."""

    state = create_initial_state("sample_docs/invoice_test_0002.jpg")
    state["document_metadata"] = {
        "file_extension": ".jpg",
        "text_length": 0,
        "has_embedded_text": False,
    }

    result = PlannerAgent(
        llm=FakePlannerLlm(FakeStructuredPlanner(error=RuntimeError("provider failed")))
    ).invoke(state)

    assert result["selected_tool"] == "OCR"
    assert "Little or no embedded text" in result["planner_reasoning"]
    assert result["logs"][-1]["metadata"]["source"] == "rules"
    assert "provider failed" in result["logs"][-1]["metadata"]["fallback_reason"]
