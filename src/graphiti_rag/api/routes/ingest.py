"""POST /api/ingest — upload documents and build the knowledge graph.

Receives a multipart/form-data list of files. Credentials live in the
process environment (set via ``POST /api/credentials``); this route does
not accept them. Files are streamed to a per-request temporary directory
under sanitised UUID names, ingested, and cleaned up.

The response carries the **original** client-supplied filenames in
``failed_files``, not the temp basenames the pipeline saw — translation
happens here so :class:`GraphAgent` can stay decoupled from HTTP concerns.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from graphiti_rag.api.models import IngestResponse
from graphiti_rag.api.state import CredentialsNotSet, app_state

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILES_PER_REQUEST = 50
MAX_FILE_BYTES = 50_000_000  # 50 MB
PIPELINE_TIMEOUT_SECONDS = 300

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/json",
        "text/plain",
    }
)

_SUFFIX_FOR_MIME: dict[str, str] = {
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/json": ".json",
    "text/plain": ".txt",
}

ALLOWED_SUFFIXES: frozenset[str] = frozenset(_SUFFIX_FOR_MIME.values())


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] = File(..., max_length=MAX_FILE_BYTES),
) -> IngestResponse:
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files — maximum {MAX_FILES_PER_REQUEST} per request",
        )
    # Hard-validate by both suffix AND MIME. Suffix is the authoritative
    # identity (it controls which loader runs server-side); MIME from the
    # browser can lie. Reject before any pipeline work happens.
    for upload in files:
        name = upload.filename or ""
        suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file extension on {name!r}: {suffix or '(none)'} — "
                    f"allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
                ),
            )
        if upload.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported MIME type on {name!r}: {upload.content_type!r}"
                ),
            )
        if _SUFFIX_FOR_MIME.get(upload.content_type) != suffix:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Extension/MIME mismatch on {name!r}: "
                    f"suffix {suffix} but MIME {upload.content_type!r}"
                ),
            )

    try:
        pipeline = await app_state.get_pipeline()
    except CredentialsNotSet as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    with tempfile.TemporaryDirectory(prefix="multi-agent-graphrag-") as tmp_dir:
        try:
            doc_paths = await _persist_uploads(files, Path(tmp_dir))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        temp_basename_to_original: dict[str, str] = {
            Path(p).name: (upload.filename or Path(p).name)
            for p, upload in zip(doc_paths, files)
        }

        try:
            result = await asyncio.wait_for(
                pipeline.ingest(doc_paths=doc_paths),
                timeout=PIPELINE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("Ingest timed out after %ds", PIPELINE_TIMEOUT_SECONDS)
            raise HTTPException(status_code=504, detail="Request timed out")
        except ValueError as exc:
            logger.warning("Ingest rejected: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid input")
        except Exception:
            logger.exception("Ingest failed")
            raise HTTPException(status_code=500, detail="Internal server error")

    failed_files = [
        temp_basename_to_original.get(b, b) for b in result.failed_files
    ]

    if result.all_failed:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All files failed to ingest.",
                "files_received": len(files),
                "successes": result.successes,
                "failures": result.failures,
                "failed_files": failed_files,
            },
        )

    if result.partially_failed:
        message = (
            f"Partial success: ingested {result.successes} of "
            f"{len(files)} document(s); {result.failures} failed."
        )
    else:
        message = f"Ingested {result.successes} document(s)."

    return IngestResponse(
        message=message,
        files_received=len(files),
        successes=result.successes,
        failures=result.failures,
        failed_files=failed_files,
    )


async def _persist_uploads(files: list[UploadFile], target_dir: Path) -> list[str]:
    paths: list[str] = []
    for upload in files:
        name = upload.filename or ""
        suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
        # Already validated in `ingest()` — re-checked here so this helper is
        # safe to call independently and never silently writes a bad suffix.
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported file extension: {suffix or '(none)'}")
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        target = target_dir / safe_name
        size = 0
        with target.open("wb") as fh:
            while True:
                chunk = await upload.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise ValueError(
                        f"File '{upload.filename}' exceeds {MAX_FILE_BYTES} bytes"
                    )
                fh.write(chunk)
        await upload.close()
        paths.append(str(target))
    return paths
