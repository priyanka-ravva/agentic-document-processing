"""Validation schemas for QA results."""

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Structured validation response from the QA Agent."""

    is_valid: bool = Field(description="Whether the extraction is acceptable.")
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
