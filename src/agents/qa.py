"""Quality assurance agent."""

from src.agents.base_agent import BaseAgent
from src.graph.state import AgentState, add_log
from src.schemas.extraction import DocumentType
from src.schemas.validation import ValidationResult


class QAAgent(BaseAgent):
    """Validates extracted structured output."""

    name = "qa"

    def invoke(self, state: AgentState) -> AgentState:
        """Validate the structured extraction output."""

        structured_output = state.get("structured_output", {})
        document_type = _normalize_document_type(state.get("document_type"))
        missing_fields: list[str] = []
        warnings: list[str] = list(structured_output.get("extraction_warnings", []))

        if document_type == DocumentType.INVOICE:
            required_fields = ["invoice_number", "invoice_date", "vendor_name", "total_amount"]
            m, w = _validate_fields(structured_output, required_fields, prefix="invoice")
            missing_fields.extend(m)
            warnings.extend(w)
        elif document_type == DocumentType.CONTRACT:
            required_fields = ["contract_title", "effective_date"]
            m, w = _validate_fields(structured_output, required_fields, prefix="contract")
            missing_fields.extend(m)
            warnings.extend(w)
            if not structured_output.get("parties"):
                missing_fields.append("contract.parties")
        elif document_type == DocumentType.MEDICAL:
            required_fields = ["patient_name", "visit_date", "provider_name"]
            m, w = _validate_fields(structured_output, required_fields, prefix="medical")
            missing_fields.extend(m)
            warnings.extend(w)
        else:
            warnings.append("Document type is unknown, so only generic validation was applied.")

        # Require zero missing fields AND zero low-confidence warnings to pass
        is_valid = bool(structured_output) and len(missing_fields) == 0 and len(warnings) == 0
        quality_score = _calculate_quality_score(missing_fields, warnings)
        validation = ValidationResult(
            is_valid=is_valid,
            missing_fields=missing_fields,
            warnings=warnings,
            quality_score=quality_score,
        )

        updated_state = state.copy()
        updated_state["validation_result"] = validation.model_dump(mode="json")

        return add_log(
            updated_state,
            agent=self.name,
            message="QA validation completed.",
            is_valid=validation.is_valid,
            missing_fields=validation.missing_fields,
            quality_score=validation.quality_score,
        )


def _validate_fields(payload: dict, fields: list[str], prefix: str) -> tuple[list[str], list[str]]:
    """Return missing fields and low-confidence warnings."""

    missing: list[str] = []
    warnings: list[str] = []
    
    for field_name in fields:
        field_payload = payload.get(field_name) or {}
        value = field_payload.get("value")
        confidence = field_payload.get("confidence", 1.0)
        
        if not value:
            missing.append(f"{prefix}.{field_name}")
        elif confidence < 0.8:
            warnings.append(f"Low confidence ({confidence}) for {prefix}.{field_name}. Please re-read the text carefully.")
            
    return missing, warnings


def _calculate_quality_score(missing_fields: list[str], warnings: list[str]) -> float:
    """Calculate a simple quality score for the current extraction."""

    score = 1.0
    score -= min(len(missing_fields) * 0.15, 0.6)
    score -= min(len(warnings) * 0.05, 0.25)
    return round(max(score, 0.0), 2)


def _normalize_document_type(document_type: DocumentType | str | None) -> DocumentType:
    """Return a supported document type enum from raw workflow state."""

    if isinstance(document_type, DocumentType):
        return document_type
    try:
        return DocumentType(document_type or DocumentType.UNKNOWN.value)
    except ValueError:
        return DocumentType.UNKNOWN
