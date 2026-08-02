"""Structured extraction schemas."""

from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    """Supported document categories."""

    INVOICE = "invoice"
    CONTRACT = "contract"
    MEDICAL = "medical"
    UNKNOWN = "unknown"


class ExtractedField(BaseModel):
    """A field extracted from the document with confidence metadata."""

    value: Optional[str] = Field(default=None, description="Extracted value as text.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: Optional[str] = Field(
        default=None,
        description="Short source phrase from the document supporting the value.",
    )


class InvoiceExtraction(BaseModel):
    """Invoice-specific structured fields."""

    invoice_number: ExtractedField = Field(default_factory=ExtractedField)
    invoice_date: ExtractedField = Field(default_factory=ExtractedField)
    due_date: ExtractedField = Field(default_factory=ExtractedField)
    vendor_name: ExtractedField = Field(default_factory=ExtractedField)
    customer_name: ExtractedField = Field(default_factory=ExtractedField)
    subtotal: ExtractedField = Field(default_factory=ExtractedField)
    tax: ExtractedField = Field(default_factory=ExtractedField)
    total_amount: ExtractedField = Field(default_factory=ExtractedField)
    currency: ExtractedField = Field(default_factory=ExtractedField)


class ContractExtraction(BaseModel):
    """Contract-specific structured fields."""

    contract_title: ExtractedField = Field(default_factory=ExtractedField)
    effective_date: ExtractedField = Field(default_factory=ExtractedField)
    parties: list[str] = Field(default_factory=list)
    term: ExtractedField = Field(default_factory=ExtractedField)
    governing_law: ExtractedField = Field(default_factory=ExtractedField)
    termination_clause: ExtractedField = Field(default_factory=ExtractedField)


class MedicalExtraction(BaseModel):
    """Medical-document-specific structured fields."""

    patient_name: ExtractedField = Field(default_factory=ExtractedField)
    visit_date: ExtractedField = Field(default_factory=ExtractedField)
    provider_name: ExtractedField = Field(default_factory=ExtractedField)
    diagnosis: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class DocumentExtraction(BaseModel):
    """Unified structured response from the Extraction Agent."""

    document_type: DocumentType = Field(description="Best classification for the document.")
    summary: str = Field(description="Brief summary of the document.")
    invoice: Optional[InvoiceExtraction] = None
    contract: Optional[ContractExtraction] = None
    medical: Optional[MedicalExtraction] = None
    additional_fields: dict[str, Any] = Field(default_factory=dict)
    extraction_warnings: list[str] = Field(default_factory=list)
