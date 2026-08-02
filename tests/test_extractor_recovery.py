"""Tests for extraction recovery normalization."""

import sys
import types

langchain_groq = types.ModuleType("langchain_groq")
langchain_groq.ChatGroq = object
sys.modules.setdefault("langchain_groq", langchain_groq)

from src.agents.extractor import (
    _chunk_extracted_text,
    _merge_chunk_outputs,
    _recover_failed_generation,
    _should_extract_in_chunks,
)
from src.schemas.extraction import ContractExtraction, DocumentExtraction, InvoiceExtraction, MedicalExtraction


def test_recovers_string_field_as_extracted_field_object() -> None:
    """String values are normalized when a schema expects ExtractedField."""

    exc = RuntimeError(
        'failed_generation: <function=InvoiceExtraction>{"invoice_number": "INV-123"}</function>'
    )

    recovered = _recover_failed_generation(exc, InvoiceExtraction)

    assert recovered["invoice_number"] == {
        "value": "INV-123",
        "confidence": 0.7,
        "evidence": None,
    }


def test_recovers_wrapped_contract_parties_as_list() -> None:
    """Contract parties may be returned as an extracted-field-like wrapper."""

    exc = RuntimeError(
        'failed_generation: <function=ContractExtraction>{"parties": {"value": ["Alpha", "Beta"], '
        '"confidence": 1.0, "evidence": "Alpha and Beta"}}</function>'
    )

    recovered = _recover_failed_generation(exc, ContractExtraction)

    assert recovered["parties"] == ["Alpha", "Beta"]


def test_recovers_unknown_document_with_minimum_defaults() -> None:
    """Unknown/generic recovery uses a real Pydantic schema, not dict."""

    exc = RuntimeError(
        'failed_generation: <function=DocumentExtraction>{"summary": "General document."}</function>'
    )

    recovered = _recover_failed_generation(exc, DocumentExtraction)

    assert recovered["document_type"] == "unknown"
    assert recovered["summary"] == "General document."


def test_repairs_provider_single_quote_escape() -> None:
    """Groq failed_generation may include invalid JSON escapes like Grave\\'s."""

    exc = RuntimeError(
        'failed_generation: <function=MedicalExtraction>{"diagnosis": ["Grave\\\'s disease"]}</function>'
    )

    recovered = _recover_failed_generation(exc, MedicalExtraction)

    assert recovered["diagnosis"] == ["Grave's disease"]


def test_large_page_marked_text_is_chunked_by_pages() -> None:
    """Large parser output is chunked without losing page markers."""

    text = "\n\n".join(f"--- Page {page} ---\ncontent {page}" for page in range(1, 12))

    assert _should_extract_in_chunks(text) is True

    chunks = _chunk_extracted_text(text, char_limit=80)

    assert len(chunks) > 1
    assert chunks[0].startswith("--- Page 1 ---")
    assert "--- Page 11 ---" in chunks[-1]


def test_chunk_outputs_merge_high_confidence_fields_and_lists() -> None:
    """Chunk outputs merge into the same flat schema shape used by QA."""

    merged = _merge_chunk_outputs(
        [
            {
                "contract_title": {"value": "NDA", "confidence": 0.6, "evidence": "NDA"},
                "parties": ["Alpha"],
            },
            {
                "contract_title": {"value": "Mutual NDA", "confidence": 0.95, "evidence": "Mutual NDA"},
                "parties": {"value": ["Alpha", "Beta"], "confidence": 0.9},
            },
        ],
        ContractExtraction,
    )

    assert merged["contract_title"]["value"] == "Mutual NDA"
    assert merged["parties"] == ["Alpha", "Beta"]
