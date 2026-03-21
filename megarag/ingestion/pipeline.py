"""
Orchestrate the full ingestion flow for a single document:
  PDF → page images + markdown → KG extraction → embeddings → stores

Each document is isolated in its own:
  - Qdrant collection: megarag_{safe_name}
  - DuckDB schema:     doc_{safe_name}
"""
import logging
import re
import time
from pathlib import Path
from tqdm import tqdm

from config.settings import get_settings
from megarag.ingestion.pdf_processor import pdf_to_images
from megarag.ingestion.doc_parser import extract_pages_markdown
from megarag.knowledge_graph.extractor import extract_entities_relations
from megarag.knowledge_graph.store import KGStore
from megarag.embedding.colqwen import ColQwenEmbedder
from megarag.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def doc_safe_name(stem: str) -> str:
    """
    Convert a PDF filename stem into a safe identifier for use as a
    Qdrant collection suffix and DuckDB schema name.

    Rules: lowercase, only [a-z0-9_], max 40 chars, no leading/trailing _.
    Example: 'My Report (2024)' → 'my_report__2024_'[:40] → 'my_report__2024'
    """
    safe = re.sub(r"[^a-z0-9_]", "_", stem.lower())[:40].strip("_")
    return safe or "doc"


def _step(label: str, start: float) -> float:
    """Log a completed step with elapsed time; return current time for next step."""
    elapsed = time.perf_counter() - start
    logger.info("[ingest] ✓ %s — %.2fs", label, elapsed)
    return time.perf_counter()


def ingest_document(pdf_path: Path) -> dict:
    """
    Full ingestion pipeline for one PDF.
    Returns a summary dict with page count, entity count, and doc_id.
    """
    cfg = get_settings()
    cfg.ensure_dirs()

    # Derive stable, safe identifiers for this document's isolated stores.
    safe = doc_safe_name(pdf_path.stem)
    collection_name = f"megarag_{safe}"
    schema = f"doc_{safe}"

    pipeline_start = time.perf_counter()
    logger.info(
        "[ingest] ═══ Starting ingestion: %s  (collection=%s  schema=%s) ═══",
        pdf_path.name, collection_name, schema,
    )

    # 1. Render page images
    logger.info("[ingest] Step 1/4 — rendering PDF pages (PyMuPDF @ 150 DPI)")
    t = time.perf_counter()
    page_images = pdf_to_images(pdf_path, cfg.pages_dir)
    t = _step(f"PDF → {len(page_images)} page image(s)", t)

    # 2. Extract per-page markdown via Docling (local, no API required)
    logger.info("[ingest] Step 2/4 — extracting per-page markdown via Docling")
    pages = extract_pages_markdown(pdf_path)
    total_chars = sum(len(p) for p in pages)
    t = _step(f"Docling extraction ({len(pages)} pages, {total_chars:,} chars)", t)

    # 3. Build knowledge graph — scoped entirely to this document's schema
    logger.info("[ingest] Step 3/4 — extracting entities & relations (%d pages)", len(pages))
    entities, relations = extract_entities_relations(pages, source=pdf_path.name)
    logger.info(
        "[ingest]   → found %d entities, %d relations", len(entities), len(relations)
    )
    kg = KGStore(cfg.kg_db_path, schema=schema)
    kg.upsert_entities(entities)
    kg.upsert_relations(relations)
    t = _step(
        f"KG extraction + store ({len(entities)} entities, {len(relations)} relations)", t
    )

    # 4. Embed page images with ColQwen → this document's own Qdrant collection
    logger.info("[ingest] Step 4/4 — embedding pages with ColQwen → Qdrant (%s)", collection_name)
    embedder = ColQwenEmbedder()
    qdrant = QdrantStore(collection_name=collection_name)

    for idx, img_path in enumerate(tqdm(page_images, desc="embedding")):
        page_start = time.perf_counter()
        embeddings = embedder.embed_page(img_path)
        qdrant.upsert_page(
            doc_name=pdf_path.name,
            page_index=idx,
            img_path=img_path,
            embeddings=embeddings,
        )
        logger.debug(
            "[ingest]   page %d/%d embedded in %.2fs",
            idx + 1, len(page_images), time.perf_counter() - page_start,
        )

    _step(f"ColQwen embedding + Qdrant upsert ({len(page_images)} pages)", t)

    total = time.perf_counter() - pipeline_start
    logger.info(
        "[ingest] ═══ Done: %s — %d pages, %d entities, %d relations | total %.2fs ═══",
        pdf_path.name, len(page_images), len(entities), len(relations), total,
    )

    return {
        "document": pdf_path.name,
        "doc_id": safe,
        "pages": len(page_images),
        "entities": len(entities),
        "relations": len(relations),
    }
