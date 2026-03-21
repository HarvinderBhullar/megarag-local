"""
Qdrant collection management and multi-vector upsert/search.
One Qdrant point = one document page.
Payload stores: doc_name, page_index, img_path (str).

Each uploaded document gets its own collection: megarag_{safe_doc_name}.
"""
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorConfig,
    MultiVectorComparator,
    VectorParams,
    PointStruct,
    ScoredPoint,
)

from config.settings import get_settings

VECTOR_NAME = "colqwen"
VECTOR_DIM = 128
COLLECTION_PREFIX = "megarag_"


class QdrantStore:
    def __init__(self, collection_name: str | None = None):
        cfg = get_settings()
        self.client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
        # Use the explicit name if provided, otherwise fall back to config default.
        self.collection = collection_name or cfg.qdrant_collection
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Class-level helpers — discover which per-doc collections exist
    # ------------------------------------------------------------------

    @classmethod
    def list_doc_collections(cls, host: str, port: int) -> list[str]:
        """Return all per-document collection names (prefix 'megarag_') in Qdrant."""
        client = QdrantClient(host=host, port=port)
        return [
            c.name
            for c in client.get_collections().collections
            if c.name.startswith(COLLECTION_PREFIX)
        ]

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    VECTOR_NAME: VectorParams(
                        size=VECTOR_DIM,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM
                        ),
                    )
                },
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_page(
        self,
        doc_name: str,
        page_index: int,
        img_path: Path,
        embeddings: list[list[float]],
    ) -> None:
        """Store one page's multi-vector embeddings with metadata."""
        point = PointStruct(
            id=str(uuid4()),
            vector={VECTOR_NAME: embeddings},
            payload={
                "doc_name": doc_name,
                "page_index": page_index,
                "img_path": str(img_path),
            },
        )
        self.client.upsert(collection_name=self.collection, points=[point])

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self, query_embeddings: list[list[float]], top_k: int = 5
    ) -> list[ScoredPoint]:
        """MaxSim search — returns top_k pages ranked by similarity."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embeddings,
            using=VECTOR_NAME,
            limit=top_k,
        )
        return results.points
