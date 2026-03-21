"""Extract structured markdown from a document using Docling (local, no API needed)."""
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

_converter = None

_PAGE_BREAK = "<!-- PageBreak -->"


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def extract_markdown(pdf_path: Path) -> str:
    """
    Convert *pdf_path* to a single markdown string using Docling.
    Runs fully locally — no API call required.
    """
    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


_FALLBACK_CHUNK_SIZE = 4_000   # chars — safe for local LLM context
_FALLBACK_CHUNK_OVERLAP = 200


def _chunk_by_chars(text: str) -> list[str]:
    """Split *text* into overlapping ~4k-char chunks, breaking on newlines."""
    chunks, start = [], 0
    while start < len(text):
        end = start + _FALLBACK_CHUNK_SIZE
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl + 1
        chunks.append(text[start:end].strip())
        start = end - _FALLBACK_CHUNK_OVERLAP
    return [c for c in chunks if c]


def extract_pages_markdown(pdf_path: Path) -> list[str]:
    """
    Convert *pdf_path* to a list of per-page markdown strings.

    Docling separates pages with ``<!-- PageBreak -->`` markers when it can
    detect page boundaries.  If no markers are present (e.g. scanned or
    single-page PDFs), the full markdown is split into ~4k-char chunks so
    the LLM never receives an oversized prompt.
    """
    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    full_md = result.document.export_to_markdown()

    # Split on Docling's page-break marker
    raw_pages = [p.strip() for p in full_md.split(_PAGE_BREAK) if p.strip()]

    if len(raw_pages) > 1:
        logger.info(
            "[doc_parser] '%s' — found %d page(s) via Docling page-break markers",
            pdf_path.name, len(raw_pages),
        )
        return raw_pages

    # No page-break markers found — fall back to character-based chunking
    logger.warning(
        "[doc_parser] '%s' — no '%s' markers found in Docling output "
        "(%d chars total); falling back to %d-char chunk splitting",
        pdf_path.name, _PAGE_BREAK, len(full_md), _FALLBACK_CHUNK_SIZE,
    )
    chunks = _chunk_by_chars(full_md)
    logger.info(
        "[doc_parser] '%s' — fallback produced %d chunk(s)", pdf_path.name, len(chunks),
    )
    return chunks
