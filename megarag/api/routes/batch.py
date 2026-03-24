"""
Batch ingestion routes.

POST /batch/ingest            — accept multiple PDFs, return job_id
GET  /batch/{job_id}/stream   — SSE stream of progress events
GET  /batch/{job_id}/status   — current status snapshot (polling fallback)
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List

import ray

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, status
from fastapi.responses import StreamingResponse

from config.settings import get_settings
from megarag.api.batch_manager import BatchJob, create_job, get_job
from megarag.api.schemas import BatchIngestResponse, BatchStatusResponse
from megarag.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/batch", tags=["batch"])


# ---------------------------------------------------------------------------
# POST /batch/ingest
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    response_model=BatchIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def batch_ingest(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
) -> BatchIngestResponse:
    """Accept one or more PDF files; start background ingestion; return job_id."""
    for f in files:
        if f.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File '{f.filename}' is not a PDF.",
            )

    # Save uploads to a temp dir that survives until the background task finishes
    tmp_dir = Path(tempfile.mkdtemp(prefix="megarag_batch_"))
    saved: list[tuple[str, Path]] = []
    for upload in files:
        dest = tmp_dir / (upload.filename or "upload.pdf")
        content = await upload.read()
        dest.write_bytes(content)
        saved.append((upload.filename or dest.name, dest))

    job = create_job([fn for fn, _ in saved])
    # Capture the main event loop so the background thread can safely post events
    main_loop = asyncio.get_event_loop()
    job.get_queue()  # create queue now, in the main loop

    background_tasks.add_task(_run_batch, job, saved, main_loop)

    return BatchIngestResponse(job_id=job.job_id, total_files=len(saved))


# ---------------------------------------------------------------------------
# GET /batch/{job_id}/stream  — SSE
# ---------------------------------------------------------------------------

@router.get("/stream/{job_id}")
async def stream_batch(job_id: str) -> StreamingResponse:
    """
    Server-Sent Events stream for a batch job.
    Clients connect and receive events until `batch_done` or `batch_failed`.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return StreamingResponse(
        _event_generator(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


async def _event_generator(job: BatchJob):
    """Yield SSE-formatted strings from the job's asyncio.Queue."""
    queue = job.get_queue()
    while True:
        try:
            event, data = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            # Send a keep-alive comment so the connection stays open
            yield ": keep-alive\n\n"
            continue

        payload = json.dumps(data)
        yield f"event: {event}\ndata: {payload}\n\n"

        if event in ("batch_done", "batch_failed"):
            break


# ---------------------------------------------------------------------------
# GET /batch/{job_id}/status  — polling fallback
# ---------------------------------------------------------------------------

@router.get("/status/{job_id}", response_model=BatchStatusResponse)
async def batch_status(job_id: str) -> BatchStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return BatchStatusResponse(**job.to_dict())


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

@ray.remote
def _ingest_remote(pdf_path_str: str) -> dict:
    """Ray remote task: run ingest_document in a worker process."""
    return ingest_document(Path(pdf_path_str))


def _run_batch(
    job: BatchJob,
    saved: list[tuple[str, Path]],
    main_loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Runs in a background thread (via BackgroundTasks).
    All files are submitted to Ray simultaneously and processed in parallel.
    SSE events are emitted as each file completes via call_soon_threadsafe.
    """
    queue = job.get_queue()

    def emit(event: str, data: dict) -> None:
        main_loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    job.overall_status = "processing"
    total = len(saved)
    failed = 0

    # Submit all files to Ray in parallel — emit file_start immediately for each
    futures: dict = {}
    for idx, (filename, pdf_path) in enumerate(saved):
        job.files[idx].status = "processing"
        emit("file_start", {
            "job_id": job.job_id,
            "filename": filename,
            "index": idx,
            "total": total,
        })
        future = _ingest_remote.remote(str(pdf_path))
        futures[future] = (idx, filename, pdf_path)

    # Collect results as they complete (in completion order, not submission order)
    pending = list(futures.keys())
    while pending:
        done, pending = ray.wait(pending, num_returns=1, timeout=None)
        future = done[0]
        idx, filename, pdf_path = futures[future]

        try:
            result = ray.get(future)
            job.files[idx].status = "done"
            job.files[idx].pages = result["pages"]
            job.files[idx].entities = result["entities"]
            job.files[idx].relations = result["relations"]

            emit("file_done", {
                "job_id": job.job_id,
                "filename": filename,
                "doc_id": result["doc_id"],
                "pages": result["pages"],
                "entities": result["entities"],
                "relations": result["relations"],
            })
        except Exception as exc:
            failed += 1
            job.files[idx].status = "failed"
            job.files[idx].error = str(exc)

            emit("file_error", {
                "job_id": job.job_id,
                "filename": filename,
                "error": str(exc),
            })
        finally:
            try:
                pdf_path.unlink(missing_ok=True)
            except Exception:
                pass

    job.overall_status = "failed" if failed == total else "done"
    emit("batch_done", {
        "job_id": job.job_id,
        "total_files": total,
        "failed": failed,
        "overall_status": job.overall_status,
    })
