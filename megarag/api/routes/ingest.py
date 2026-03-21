import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from megarag.api.schemas import IngestResponse
from megarag.ingestion.pipeline import ingest_document

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / file.filename
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        result = ingest_document(tmp_path)

    return IngestResponse(**result)
