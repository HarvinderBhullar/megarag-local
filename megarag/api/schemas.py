from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document: str
    doc_id: str          # safe identifier — use this for scoped queries
    pages: int
    entities: int
    relations: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    # Optional: scope the query to a single document.
    # Use the 'doc_id' returned by /ingest.  If omitted, all documents are searched.
    doc_id: Optional[str] = Field(default=None, description="doc_id from /ingest response")


class QueryResponse(BaseModel):
    question: str
    answer: str
    draft: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Batch ingestion schemas
# ---------------------------------------------------------------------------

class BatchIngestResponse(BaseModel):
    job_id: str
    total_files: int


class FileStatusModel(BaseModel):
    filename: str
    status: Literal["pending", "processing", "done", "failed"]
    pages: int = 0
    entities: int = 0
    relations: int = 0
    error: Optional[str] = None


class BatchStatusResponse(BaseModel):
    job_id: str
    overall_status: Literal["pending", "processing", "done", "failed"]
    created_at: str
    files: List[FileStatusModel]


# ---------------------------------------------------------------------------
# Knowledge graph schema
# ---------------------------------------------------------------------------

class KGGraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]  # [{"data": {"id": ..., "label": ..., "type": ...}}]
    edges: List[Dict[str, Any]]  # [{"data": {"id": ..., "source": ..., "target": ..., "label": ...}}]
