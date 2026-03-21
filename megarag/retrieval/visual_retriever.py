"""Retrieve top-K page images using ColQwen query embeddings + Qdrant MaxSim."""
from pathlib import Path

from config.settings import get_settings
from megarag.embedding.colqwen import ColQwenEmbedder
from megarag.vectorstore.qdrant_store import QdrantStore


def retrieve_pages(
    question: str,
    top_k: int = 5,
    collection_name: str | None = None,
) -> list[dict]:
    """
    Embed *question* with ColQwen and run MaxSim search in Qdrant.

    If *collection_name* is given, search only that document's collection.
    Otherwise fan-out across all per-document collections (prefix 'megarag_'),
    merge results, and return the global top-k.

    Returns a list of page metadata dicts (doc_name, page_index, img_path, score).
    """
    cfg = get_settings()
    embedder = ColQwenEmbedder()
    query_vecs = embedder.embed_query(question)

    if collection_name:
        collections = [collection_name]
    else:
        collections = QdrantStore.list_doc_collections(cfg.qdrant_host, cfg.qdrant_port)

    if not collections:
        return []

    all_hits = []
    for coll in collections:
        qdrant = QdrantStore(collection_name=coll)
        hits = qdrant.search(query_vecs, top_k=top_k)
        all_hits.extend(hits)

    # Merge and keep global top-k ranked by score.
    all_hits.sort(key=lambda h: h.score, reverse=True)
    all_hits = all_hits[:top_k]

    return [
        {
            "doc_name": h.payload["doc_name"],
            "page_index": h.payload["page_index"],
            "img_path": Path(h.payload["img_path"]),
            "score": h.score,
        }
        for h in all_hits
    ]
