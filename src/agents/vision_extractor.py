"""Vision-based extraction agent for fallback OCR retries."""

import base64
from pathlib import Path

import fitz
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.config import get_llm, get_settings
from src.graph.state import AgentState, add_log
from src.prompts.prompt_factory import PromptFactory
from src.schemas.extraction import (
    ContractExtraction,
    DocumentExtraction,
    DocumentType,
    InvoiceExtraction,
    MedicalExtraction,
)

# Groq models known to accept image input, per the Groq /models API's
# "input_modalities" field. GROQ_MODEL/GROQ_FALLBACK_MODELS are configured for
# text tasks and are not assumed to support vision unless listed here.
VISION_CAPABLE_MODELS = {"qwen/qwen3.6-27b"}


class VisionExtractionAgent(BaseAgent):
    """Extracts structured information from document images using a Vision LLM."""

    name = "vision_extractor"

    def invoke(self, state: AgentState) -> AgentState:
        """Extract structured document fields using a Vision LLM directly on the image."""

        document_type = _normalize_document_type(state.get("document_type"))
        file_path = state.get("file_path", "")
        updated_state = state.copy()

        if not file_path or not Path(file_path).exists():
            fallback = DocumentExtraction(
                document_type=DocumentType.UNKNOWN,
                summary="Vision extraction failed. Image file not found.",
                extraction_warnings=["File path missing or invalid."],
            )
            updated_state["structured_output"] = fallback.model_dump(mode="json")
            return add_log(
                updated_state,
                agent=self.name,
                message="Vision Extractor skipped because file was missing.",
            )

        encoded_string, mime_type, image_note = _encode_file_for_vision(file_path)
        if not encoded_string:
            previous_output = state.get("previous_structured_output") or {}
            if previous_output:
                restored_output = dict(previous_output)
                warnings = list(restored_output.get("extraction_warnings", []))
                warnings.append(image_note)
                restored_output["extraction_warnings"] = warnings
                updated_state["structured_output"] = restored_output
                updated_state["document_type"] = state.get(
                    "previous_document_type"
                ) or _document_type_value(
                    document_type
                )
                updated_state["force_finalize"] = True
            else:
                fallback = DocumentExtraction(
                    document_type=DocumentType.UNKNOWN,
                    summary="Vision extraction skipped because the file was too large for direct vision fallback.",
                    extraction_warnings=[image_note],
                )
                updated_state["structured_output"] = fallback.model_dump(mode="json")
            return add_log(
                updated_state,
                agent=self.name,
                message="Vision extraction skipped before provider call.",
                reason=image_note,
                restored_previous_output=bool(previous_output),
            )

        # Determine specific prompt and schema based on classification
        schema = DocumentExtraction
        prompt_name = "extractor"

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

        system_prompt += "\n\nCRITICAL: The previous OCR extraction completely failed or returned garbled text. You are a Multimodal Vision model. Please carefully inspect the pixels of the attached image to read the correct values."

        # Try vision-capable fallback models first. Rate limits are not retried
        # because the provider has already told us more calls will fail.
        settings = get_settings()
        configured_models = [
            model.strip()
            for model in [*settings.groq_fallback_models.split(","), settings.groq_model]
            if model.strip()
        ]
        # Only send image payloads to models that actually accept image input.
        # GROQ_MODEL/GROQ_FALLBACK_MODELS are chosen for text classification/extraction
        # and may not support vision (e.g. llama-3.1-8b-instant is text-only), so a
        # configured text-only model must never be attempted here.
        vision_model_names = [
            model for model in dict.fromkeys(configured_models) if model in VISION_CAPABLE_MODELS
        ]
        if not vision_model_names:
            vision_model_names = [settings.groq_model]

        errors: list[str] = []
        for vision_model_name in vision_model_names[:2]:
            llm = get_llm(model_name=vision_model_name)
            structured_llm = llm.with_structured_output(schema)

            message_content = [
                {"type": "text", "text": f"Extract structured data directly from this document image. {image_note}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                }
            ]

            try:
                response = structured_llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=message_content),
                    ]
                )
            except Exception as exc:
                errors.append(f"{vision_model_name}: {exc}")
                if _is_rate_limit_error(exc):
                    break
                continue

            if isinstance(response, BaseModel):
                structured_output = response.model_dump(mode="json")
            else:
                structured_output = dict(response)

            document_type_value = _document_type_value(document_type)
            structured_output.setdefault("document_type", document_type_value)
            updated_state["document_type"] = document_type_value
            updated_state["structured_output"] = structured_output

            return add_log(
                updated_state,
                agent=self.name,
                message=f"Multimodal Vision extraction completed using {vision_model_name}.",
                document_type=document_type_value,
                model=vision_model_name,
            )

        previous_output = state.get("previous_structured_output") or {}
        if previous_output:
            restored_output = dict(previous_output)
            warnings = list(restored_output.get("extraction_warnings", []))
            warnings.append(
                "Vision fallback failed; preserved OCR/text extraction result. "
                f"Errors: {' | '.join(errors)}"
            )
            restored_output["extraction_warnings"] = warnings
            updated_state["structured_output"] = restored_output
            updated_state["document_type"] = state.get(
                "previous_document_type"
            ) or _document_type_value(document_type)
            updated_state["force_finalize"] = True
        else:
            fallback = DocumentExtraction(
                document_type=DocumentType.UNKNOWN,
                summary=f"Vision extraction failed: {' | '.join(errors)}",
                extraction_warnings=errors,
            )
            updated_state["structured_output"] = fallback.model_dump(mode="json")
        return add_log(
            updated_state,
            agent=self.name,
            message="Vision extraction failed; restored previous OCR/text result when available.",
            errors=errors,
            restored_previous_output=bool(previous_output),
        )


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True when a provider error is clearly a rate-limit response."""

    error_text = str(exc).lower()
    return "rate_limit" in error_text or "rate limit" in error_text or "429" in error_text


def _normalize_document_type(document_type: DocumentType | str | None) -> DocumentType:
    """Return a supported document type enum from raw workflow state."""

    if isinstance(document_type, DocumentType):
        return document_type
    try:
        return DocumentType(document_type or DocumentType.UNKNOWN.value)
    except ValueError:
        return DocumentType.UNKNOWN


def _encode_file_for_vision(file_path: str, max_base64_chars: int = 4_000_000) -> tuple[str, str, str]:
    """Encode an image, or render the first PDF page, for vision fallback."""

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        with fitz.open(path) as document:
            if document.page_count == 0:
                return "", "image/png", "PDF had no pages to render for vision fallback."
            page = document[0]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            png_bytes = pixmap.tobytes("png")
        encoded_string = base64.b64encode(png_bytes).decode("utf-8")
        if len(encoded_string) > max_base64_chars:
            return "", "image/png", "Rendered first PDF page was too large for vision fallback."
        return encoded_string, "image/png", "Only the first PDF page was rendered for vision fallback."

    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    if len(encoded_string) > max_base64_chars:
        return "", "image/jpeg", "Image payload was too large for vision fallback."

    mime_type = "image/jpeg"
    if suffix == ".png":
        mime_type = "image/png"
    return encoded_string, mime_type, "Original image was used for vision fallback."


def _document_type_value(document_type: DocumentType | str) -> str:
    """Normalize enum/string document types for trace output."""

    if isinstance(document_type, DocumentType):
        return document_type.value
    return document_type or DocumentType.UNKNOWN.value
