"""Extraction agent."""

import json
import re
from typing import get_origin

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.config import get_llm, get_model_names
from src.graph.state import AgentState, add_log
from src.prompts.prompt_factory import PromptFactory
from src.schemas.extraction import (
    ContractExtraction,
    DocumentExtraction,
    DocumentType,
    ExtractedField,
    InvoiceExtraction,
    MedicalExtraction,
)

LARGE_DOCUMENT_PAGE_THRESHOLD = 10
LARGE_DOCUMENT_CHAR_THRESHOLD = 30000
CHUNK_CHAR_LIMIT = 12000


class ExtractionAgent(BaseAgent):
    """Extracts structured information from document text."""

    name = "extractor"

    def invoke(self, state: AgentState) -> AgentState:
        """Extract structured document fields """

        extracted_text = state.get("extracted_text", "").strip()
        document_type = _normalize_document_type(state.get("document_type"))
        updated_state = state.copy()

        if not extracted_text:
            fallback = DocumentExtraction(
                document_type=DocumentType.UNKNOWN,
                summary="No text was extracted from the document.",
                extraction_warnings=["Extraction skipped because extracted_text was empty."],
            )
            updated_state["structured_output"] = fallback.model_dump(mode="json")
            return add_log(
                updated_state,
                agent=self.name,
                message="Extractor skipped because no text was available.",
            )

        # Determine specific prompt and schema based on classification
        schema = DocumentExtraction
        prompt_name = self.name
        
        if document_type == DocumentType.INVOICE:
            prompt_name = "extractor_invoice"
            schema = InvoiceExtraction
        elif document_type == DocumentType.CONTRACT:
            prompt_name = "extractor_contract"
            schema = ContractExtraction
        elif document_type == DocumentType.MEDICAL:
            prompt_name = "extractor_medical"
            schema = MedicalExtraction

        prompt = PromptFactory.get_prompt(agent=prompt_name, context=state)
        system_prompt = prompt.system_prompt

        retry_count = state.get("retry_count", 0)
        if retry_count > 0:
            validation_result = state.get("validation_result", {})
            system_prompt += "\n\nPREVIOUS ATTEMPT FAILED VALIDATION. Please fix the following errors:\n"
            if missing_fields := validation_result.get("missing_fields"):
                system_prompt += f"- Missing fields: {', '.join(missing_fields)}\n"
            if warnings := validation_result.get("warnings"):
                system_prompt += f"- Warnings: {', '.join(warnings)}\n"

        if self.llm:
            return self._extract_with_llm(state, updated_state, extracted_text, system_prompt, schema, self.llm, "injected")

        errors: list[str] = []
        for model_name in get_model_names():
            try:
                llm = get_llm(model_name=model_name)
                return self._extract_with_llm(
                    state,
                    updated_state,
                    extracted_text,
                    system_prompt,
                    schema,
                    llm,
                    model_name,
                )
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")

        error_message = " | ".join(errors)
        fallback = DocumentExtraction(
            document_type=DocumentType.UNKNOWN,
            summary="Structured extraction failed for all configured Groq models.",
            additional_fields={"text_preview": extracted_text[:1000]},
            extraction_warnings=errors,
        )
        updated_state["structured_output"] = fallback.model_dump(mode="json")
        updated_state["error"] = error_message
        return add_log(
            updated_state,
            agent=self.name,
            message="Structured extraction failed for all configured models; fallback output was created.",
            errors=errors,
        )

    def _extract_with_llm(
        self,
        state: AgentState,
        updated_state: AgentState,
        extracted_text: str,
        system_prompt: str,
        schema,
        llm,
        model_name: str,) -> AgentState:
        """Run structured extraction with one configured model."""

        try:
            structured_llm = llm.with_structured_output(schema)
            if _should_extract_in_chunks(extracted_text):
                chunk_outputs = []
                for chunk in _chunk_extracted_text(extracted_text):
                    chunk_outputs.append(
                        _invoke_structured_with_recovery(
                            structured_llm=structured_llm,
                            messages=[
                                SystemMessage(content=system_prompt),
                                HumanMessage(content=f"Extract structured data from this document text chunk:\n\n{chunk}"),
                            ],
                            schema=schema,
                        )
                    )

                updated_state["structured_output"] = _merge_chunk_outputs(chunk_outputs, schema)
                return add_log(
                    updated_state,
                    agent=self.name,
                    message="Structured extraction completed with chunked large-document processing.",
                    document_type=_document_type_value(updated_state.get("document_type")),
                    model=model_name,
                    chunk_count=len(chunk_outputs),
                )

            updated_state["structured_output"] = _invoke_structured_with_recovery(
                structured_llm=structured_llm,
                messages=[
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Extract structured data from this document text:\n\n{extracted_text}"),
                ],
                schema=schema,
            )
            return add_log(
                updated_state,
                agent=self.name,
                message="Structured extraction completed.",
                document_type=_document_type_value(updated_state.get("document_type")),
                model=model_name,
            )
        except Exception as exc:
            raise RuntimeError(f"Structured extraction failed with {model_name}: {exc}") from exc


def _invoke_structured_with_recovery(structured_llm, messages: list, schema) -> dict:
    """Invoke a structured LLM and recover provider failed_generation payloads."""

    try:
        return _to_json_dict(structured_llm.invoke(messages))
    except Exception as exc:
        recovered_output = _recover_failed_generation(exc, schema)
        if recovered_output:
            return recovered_output
        raise


def _to_json_dict(response: BaseModel | dict) -> dict:
    """Normalize structured LLM responses to a JSON-safe dictionary."""

    if isinstance(response, BaseModel):
        return response.model_dump(mode="json")
    return response


def _recover_failed_generation(exc: Exception, schema) -> dict:
    """Recover valid JSON from provider tool-call errors when possible."""

    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        return {}

    error_text = str(exc)
    if "failed_generation" not in error_text:
        return {}

    match = re.search(r"<function=[^>]+>\s*(\{.*\})", error_text, flags=re.DOTALL)
    if not match:
        return {}

    json_text = match.group(1)
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(_repair_provider_json(json_text))
        payload = _normalize_payload_for_schema(payload, schema)
        return schema.model_validate(payload).model_dump(mode="json")
    except (json.JSONDecodeError, ValueError):
        return {}


def _repair_provider_json(json_text: str) -> str:
    """Repair common provider failed_generation JSON escaping issues."""

    return json_text.replace("\\'", "'")


def _should_extract_in_chunks(extracted_text: str) -> bool:
    """Return true when searchable PDF text is large enough to chunk."""

    pages = _split_extracted_pages(extracted_text)
    return len(pages) > LARGE_DOCUMENT_PAGE_THRESHOLD or len(extracted_text) > LARGE_DOCUMENT_CHAR_THRESHOLD


def _split_extracted_pages(extracted_text: str) -> list[str]:
    """Split parser output into page-sized strings using PDF parser markers."""

    matches = list(re.finditer(r"(?m)^--- Page \d+ ---$", extracted_text))
    if not matches:
        return [extracted_text] if extracted_text.strip() else []

    pages: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(extracted_text)
        page_text = extracted_text[start:end].strip()
        if page_text:
            pages.append(page_text)
    return pages


def _chunk_extracted_text(extracted_text: str, char_limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """Group extracted PDF pages into chunks bounded by approximate size."""

    pages = _split_extracted_pages(extracted_text)
    chunks: list[str] = []
    current_pages: list[str] = []
    current_length = 0

    for page in pages:
        page_length = len(page)
        if current_pages and current_length + page_length > char_limit:
            chunks.append("\n\n".join(current_pages))
            current_pages = []
            current_length = 0

        current_pages.append(page)
        current_length += page_length

    if current_pages:
        chunks.append("\n\n".join(current_pages))

    return chunks


def _merge_chunk_outputs(chunk_outputs: list[dict], schema) -> dict:
    """Merge chunk-level structured outputs into one schema-compatible payload."""

    if not chunk_outputs:
        return {}

    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        return chunk_outputs[-1]

    merged: dict = {}
    for field_name, field_info in schema.model_fields.items():
        annotation = field_info.annotation
        values = [output.get(field_name) for output in chunk_outputs if isinstance(output, dict)]

        if annotation is ExtractedField:
            merged[field_name] = _best_extracted_field(values)
        elif get_origin(annotation) is list:
            merged[field_name] = _merge_list_values(values)
        elif field_name == "additional_fields":
            merged[field_name] = _merge_dict_values(values)
        elif field_name == "extraction_warnings":
            merged[field_name] = _merge_list_values(values)
        elif field_name == "summary":
            merged[field_name] = _merge_summary(values)
        elif field_name == "document_type":
            merged[field_name] = _merge_document_type(values)
        else:
            merged[field_name] = next((value for value in values if value not in (None, "", [], {})), None)

    return schema.model_validate(_normalize_payload_for_schema(merged, schema)).model_dump(mode="json")


def _best_extracted_field(values: list) -> dict:
    """Pick the highest-confidence non-empty extracted field from chunks."""

    best = {"value": None, "confidence": 0.0, "evidence": None}
    for value in values:
        field = _normalize_extracted_field(value)
        if not field.get("value"):
            continue
        if field.get("confidence", 0.0) > best.get("confidence", 0.0):
            best = field
    return best


def _merge_list_values(values: list) -> list:
    """Merge list-like chunk values while preserving order."""

    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _normalize_list_field(value):
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged


def _merge_dict_values(values: list) -> dict:
    """Merge dict values from chunks."""

    merged: dict = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _merge_summary(values: list) -> str:
    """Keep a compact combined summary from chunk summaries."""

    summaries = [str(value).strip() for value in values if value]
    return " ".join(summaries[:3])


def _merge_document_type(values: list) -> str:
    """Return the first supported non-empty document type."""

    for value in values:
        if value:
            return str(value)
    return DocumentType.UNKNOWN.value


def _normalize_payload_for_schema(payload: dict, schema: type[BaseModel]) -> dict:
    """Coerce common LLM shape mistakes before Pydantic validation."""

    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)

    if schema is DocumentExtraction:
        normalized.setdefault("document_type", DocumentType.UNKNOWN.value)
        normalized.setdefault("summary", "")

    for field_name, field_info in schema.model_fields.items():
        if field_name not in normalized:
            continue

        annotation = field_info.annotation
        field_value = normalized[field_name]

        if annotation is ExtractedField:
            normalized[field_name] = _normalize_extracted_field(field_value)
        elif get_origin(annotation) is list:
            normalized[field_name] = _normalize_list_field(field_value)

    return normalized


def _normalize_extracted_field(value) -> dict:
    """Return an ExtractedField-compatible object for raw scalar values."""

    if isinstance(value, dict):
        return {
            "value": value.get("value"),
            "confidence": value.get("confidence", 0.7),
            "evidence": value.get("evidence"),
        }

    if value is None:
        return {"value": None, "confidence": 0.0, "evidence": None}

    return {"value": str(value), "confidence": 0.7, "evidence": None}


def _normalize_list_field(value) -> list:
    """Return a list for schema fields where models sometimes return wrappers."""

    if isinstance(value, dict):
        value = value.get("value", [])

    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value if item is not None]

    return [str(value)]


def _document_type_value(document_type: DocumentType | str | None) -> str:
    """Normalize document type values for trace metadata."""

    if isinstance(document_type, DocumentType):
        return document_type.value
    return document_type or DocumentType.UNKNOWN.value


def _normalize_document_type(document_type: DocumentType | str | None) -> DocumentType:
    """Return a supported document type enum from raw workflow state."""

    if isinstance(document_type, DocumentType):
        return document_type
    try:
        return DocumentType(document_type or DocumentType.UNKNOWN.value)
    except ValueError:
        return DocumentType.UNKNOWN
