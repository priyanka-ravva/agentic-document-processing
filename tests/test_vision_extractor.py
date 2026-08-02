"""Tests for vision fallback file preparation."""

import sys
import types

import fitz

langchain_groq = types.ModuleType("langchain_groq")
langchain_groq.ChatGroq = object
sys.modules.setdefault("langchain_groq", langchain_groq)

from src.agents.vision_extractor import _encode_file_for_vision, _normalize_document_type
from src.schemas.extraction import DocumentType


def test_pdf_is_rendered_to_png_for_vision_fallback(tmp_path) -> None:
    """PDF vision fallback should send an image, not the raw PDF bytes."""

    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello")
    document.save(pdf_path)
    document.close()

    encoded, mime_type, note = _encode_file_for_vision(str(pdf_path))

    assert encoded
    assert mime_type == "image/png"
    assert "first PDF page" in note


def test_document_type_string_is_normalized_for_vision_schema_selection() -> None:
    """Workflow state stores document_type as a string after classification."""

    assert _normalize_document_type("invoice") == DocumentType.INVOICE
    assert _normalize_document_type("contract") == DocumentType.CONTRACT
    assert _normalize_document_type("medical") == DocumentType.MEDICAL
    assert _normalize_document_type("unexpected") == DocumentType.UNKNOWN
