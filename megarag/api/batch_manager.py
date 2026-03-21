"""
In-memory batch job manager.

Each batch job tracks the ingestion status of multiple PDFs and exposes
an asyncio.Queue that the SSE endpoint reads from to stream progress events.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional


FileStatusLiteral = Literal["pending", "processing", "done", "failed"]
JobStatusLiteral = Literal["pending", "processing", "done", "failed"]


@dataclass
class FileStatus:
    filename: str
    status: FileStatusLiteral = "pending"
    pages: int = 0
    entities: int = 0
    relations: int = 0
    error: Optional[str] = None


@dataclass
class BatchJob:
    job_id: str
    files: List[FileStatus]
    overall_status: JobStatusLiteral = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    # asyncio.Queue for SSE events; created lazily so it lives in the right loop
    _queue: Optional[asyncio.Queue] = field(default=None, repr=False, compare=False)

    def get_queue(self) -> asyncio.Queue:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "overall_status": self.overall_status,
            "created_at": self.created_at.isoformat(),
            "files": [
                {
                    "filename": f.filename,
                    "status": f.status,
                    "pages": f.pages,
                    "entities": f.entities,
                    "relations": f.relations,
                    "error": f.error,
                }
                for f in self.files
            ],
        }


# Module-level job store — shared across all requests in a single process
_jobs: Dict[str, BatchJob] = {}


def create_job(filenames: List[str]) -> BatchJob:
    job_id = uuid.uuid4().hex
    job = BatchJob(
        job_id=job_id,
        files=[FileStatus(filename=fn) for fn in filenames],
    )
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[BatchJob]:
    return _jobs.get(job_id)
