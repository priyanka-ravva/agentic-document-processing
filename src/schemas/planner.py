"""Schemas for planner decisions."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ExtractionTool(StrEnum):
    """Supported extraction tools."""

    PDF_PARSER = "PDF_PARSER"
    OCR = "OCR"
    TEXT_PARSER = "TEXT_PARSER"
    VISION_LLM = "VISION_LLM"


class PlannerDecision(BaseModel):
    """Structured output from the Planner Agent."""

    selected_tool: ExtractionTool = Field(description="Tool selected for text extraction.")
    reasoning: str = Field(description="Concise reason for the selected tool.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0 to 1.")
