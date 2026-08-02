"""Tests for extraction recovery normalization."""

import sys
import types

langchain_groq = types.ModuleType("langchain_groq")
langchain_groq.ChatGroq = object
sys.modules.setdefault("langchain_groq", langchain_groq)

from src.agents.extractor import (
    _chunk_extracted_text,
    _fallback_extract_invoice_fields,
    _is_structured_output_empty,
    _merge_chunk_outputs,
    _recover_failed_generation,
    _repair_provider_json,
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


def test_repairs_provider_function_closure_trailing_text() -> None:
    """Groq failed_generation payloads may include trailing tool wrapper text."""

    raw = '{"invoice_number": "INV-123"} </function>'
    assert _repair_provider_json(raw) == '{"invoice_number": "INV-123"}'


def test_empty_structured_output_is_detected() -> None:
    """Structured outputs with no extracted values are considered empty."""

    output = {
        "invoice_number": {"value": None, "confidence": 0.0, "evidence": None},
        "subtotal": {"value": None, "confidence": 0.0, "evidence": None},
    }

    assert _is_structured_output_empty(output) is True
    assert _is_structured_output_empty({"invoice_number": {"value": "INV-1", "confidence": 1.0, "evidence": "INV-1"}}) is False


def test_fallback_extracts_invoice_fields_from_plain_text() -> None:
    """Fallback extraction can parse invoice information from plain text."""

    text = (
        "INVOICE\n\n"
        "From: TechParts Wholesale Inc.\n"
        "To: MetroOffice Supplies Co.\n"
        "Invoice No: TW-2024-0562\n"
        "Date: October 5, 2024\n"
        "Due Date: November 4, 2024\n"
        "Subtotal: $3,775.00\n"
        "Tax (7%): $264.25\n"
        "TOTAL DUE: $4,039.25\n"
        "Currency: USD\n"
    )

    recovered = _fallback_extract_invoice_fields(text)

    assert recovered["invoice_number"]["value"] == "TW-2024-0562"
    assert recovered["invoice_date"]["value"] == "October 5, 2024"
    assert recovered["due_date"]["value"] == "November 4, 2024"
    assert recovered["vendor_name"]["value"] == "TechParts Wholesale Inc."
    assert recovered["customer_name"]["value"] == "MetroOffice Supplies Co."
    assert recovered["subtotal"]["value"] == "3,775.00"
    assert recovered["tax"]["value"] == "264.25"
    assert recovered["total_amount"]["value"] == "4,039.25"
    assert recovered["currency"]["value"] == "USD"


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
