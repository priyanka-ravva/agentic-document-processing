"""Document inspection utilities."""

from pathlib import Path
from typing import Any

import fitz


def analyze_document(file_path: str) -> dict[str, Any]:
    """Inspect a document and return routing metadata."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    supported_text_extensions = {".txt", ".csv", ".json", ".xlsx", ".docx"}
    metadata: dict[str, Any] = {
        "file_name": path.name,
        "file_extension": path.suffix.lower(),
        "file_size_bytes": path.stat().st_size,
        "page_count": 0,
        "text_length": 0,
        "has_embedded_text": False,
        "image_count": 0,
        "is_pdf": path.suffix.lower() == ".pdf",
        "is_text_document": path.suffix.lower() in supported_text_extensions,
    }

    if not metadata["is_pdf"]:
        return metadata

    with fitz.open(path) as document:
        metadata["page_count"] = document.page_count
        for page in document:
            page_text = page.get_text("text").strip()
            metadata["text_length"] += len(page_text)
            metadata["image_count"] += len(page.get_images(full=True))

    metadata["has_embedded_text"] = metadata["text_length"] > 0
    return metadata
