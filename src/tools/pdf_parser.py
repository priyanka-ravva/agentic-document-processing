"""PDF text extraction tool."""

from pathlib import Path

import fitz


def extract_text_from_pdf(file_path: str) -> str:
    """Extract embedded text from a PDF using PyMuPDF."""

    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF parser only supports .pdf files. Got: {path.suffix}")

    pages: list[str] = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"--- Page {page_number} ---\n{text}")

    return "\n\n".join(pages)
