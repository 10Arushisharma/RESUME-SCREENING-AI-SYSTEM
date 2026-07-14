from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz
from PIL import Image
from docx import Document as DocxDocument

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

logger = logging.getLogger(__name__)


class PDFService:
    """Service responsible for extracting text from common resume formats."""

    def extract_text(self, file_path: Path | str, file_name: str | None = None) -> str:
        path = Path(file_path)
        if not path.exists():
            logger.warning("File path does not exist: %s", path)
            return ""

        return self.extract_text_from_bytes(path.read_bytes(), file_name=file_name or path.name)

    def extract_text_from_bytes(self, content: bytes, file_name: str | None = None) -> str:
        file_name = (file_name or "resume").lower()

        if file_name.endswith(".pdf"):
            return self._extract_pdf_text(content, file_name)
        if file_name.endswith((".doc", ".docx")):
            return self._extract_docx_text(content, file_name)
        if file_name.endswith((".txt", ".rtf", ".md")):
            return self._decode_text_bytes(content)
        if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            return self._extract_image_text(content, file_name)
        return self._decode_text_bytes(content)

    def _extract_pdf_text(self, content: bytes, file_name: str) -> str:
        try:
            with fitz.open(stream=content, filetype="pdf") as document:
                if document.page_count == 0:
                    logger.warning("PDF %s is empty", file_name)
                    return ""
                pages: list[str] = []
                for page_number in range(document.page_count):
                    page = document.load_page(page_number)
                    pages.append(page.get_text("text") or "")
                return "\n".join(page for page in pages if page).strip()
        except (RuntimeError, ValueError, fitz.FileDataError) as exc:
            logger.exception("Failed to extract text from PDF %s", file_name)
            return ""

    def _extract_docx_text(self, content: bytes, file_name: str) -> str:
        try:
            import io

            document = DocxDocument(io.BytesIO(content))
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            return "\n".join(paragraphs).strip()
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to extract text from DOCX %s", file_name)
            return ""

    def _decode_text_bytes(self, content: bytes) -> str:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="ignore")

    def _extract_image_text(self, content: bytes, file_name: str) -> str:
        try:
            image = Image.open(BytesIO(content))
            image = image.convert("RGB")
            if pytesseract is None:
                return ""
            return pytesseract.image_to_string(image)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to extract text from image %s", file_name)
            return ""
