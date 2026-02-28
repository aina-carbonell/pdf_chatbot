"""
Document Processor
Extracts and chunks text from PDF, DOCX, TXT, MD files.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extracts text from various document types and splits into chunks."""

    CHUNK_SIZE = 800       # characters per chunk (roughly 200 tokens)
    CHUNK_OVERLAP = 150    # overlap to preserve context between chunks

    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a document and return a list of text chunks.
        Each chunk: {text, page, chunk_index, section}
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            pages = self._extract_pdf(path)
        elif ext in (".docx", ".doc"):
            pages = self._extract_docx(path)
        elif ext in (".txt", ".md"):
            pages = self._extract_text(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = self._chunk_pages(pages)
        logger.info(f"Processed {path.name}: {len(pages)} pages → {len(chunks)} chunks")
        return chunks

    # ── Extractors ────────────────────────────────────────────────────────────

    def _extract_pdf(self, path: Path) -> List[Dict]:
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({"page": i, "text": text})
            return pages
        except ImportError:
            logger.warning("pdfplumber not installed, trying PyPDF2")
            return self._extract_pdf_fallback(path)

    def _extract_pdf_fallback(self, path: Path) -> List[Dict]:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i, "text": text})
            return pages
        except ImportError:
            raise ImportError("Please install pdfplumber or PyPDF2: pip install pdfplumber")

    def _extract_docx(self, path: Path) -> List[Dict]:
        try:
            import docx
            doc = docx.Document(str(path))
            # Group paragraphs into virtual pages (~30 paragraphs each)
            page_size = 30
            all_paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            pages = []
            for i in range(0, len(all_paragraphs), page_size):
                chunk = all_paragraphs[i:i + page_size]
                pages.append({
                    "page": (i // page_size) + 1,
                    "text": "\n".join(chunk)
                })
            return pages
        except ImportError:
            raise ImportError("Please install python-docx: pip install python-docx")

    def _extract_text(self, path: Path) -> List[Dict]:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Split into virtual pages of ~2000 chars
        page_size = 2000
        pages = []
        for i in range(0, len(text), page_size):
            snippet = text[i:i + page_size]
            if snippet.strip():
                pages.append({"page": (i // page_size) + 1, "text": snippet})
        return pages

    # ── Chunker ───────────────────────────────────────────────────────────────

    def _chunk_pages(self, pages: List[Dict]) -> List[Dict[str, Any]]:
        """Split page texts into overlapping chunks preserving page metadata."""
        chunks = []
        chunk_index = 0

        for page_data in pages:
            page_num = page_data["page"]
            text = self._clean_text(page_data["text"])

            # Split into sentences then group into chunks
            sentences = self._split_sentences(text)
            buffer = ""

            for sentence in sentences:
                if len(buffer) + len(sentence) <= self.CHUNK_SIZE:
                    buffer += (" " if buffer else "") + sentence
                else:
                    if buffer:
                        chunks.append({
                            "text": buffer.strip(),
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "section": self._detect_section(buffer)
                        })
                        chunk_index += 1
                        # Overlap: keep last N chars
                        buffer = buffer[-self.CHUNK_OVERLAP:] + " " + sentence
                    else:
                        buffer = sentence

            if buffer.strip():
                chunks.append({
                    "text": buffer.strip(),
                    "page": page_num,
                    "chunk_index": chunk_index,
                    "section": self._detect_section(buffer)
                })
                chunk_index += 1

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _detect_section(self, text: str) -> str:
        """Try to detect the heading/section of a chunk."""
        lines = text.strip().split("\n")
        first_line = lines[0].strip() if lines else ""
        if len(first_line) < 80 and first_line.endswith((":", ".")):
            return first_line
        return ""