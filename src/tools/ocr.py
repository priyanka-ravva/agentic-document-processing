"""OCR extraction tool."""

from pathlib import Path

import fitz
import pytesseract
from PIL import Image


def extract_text_with_ocr(file_path: str, dpi: int = 200) -> str:
    """Extract text from a scanned PDF or image using Tesseract OCR."""

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return _ocr_pdf(path, dpi=dpi)

    if extension in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return pytesseract.image_to_string(Image.open(path)).strip()

    raise ValueError(f"OCR does not support this file type yet: {extension}")


def _ocr_pdf(path: Path, dpi: int) -> str:
    """Render PDF pages to images and OCR each page."""

    page_texts: list[str] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            text = pytesseract.image_to_string(image).strip()
            if text:
                page_texts.append(f"--- Page {page_number} ---\n{text}")

    return "\n\n".join(page_texts)
