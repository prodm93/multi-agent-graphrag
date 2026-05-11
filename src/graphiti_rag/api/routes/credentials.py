"""Credential routes.

* ``POST /api/credentials/parse`` — accepts a Neo4j Aura ``.txt`` upload and
  returns the parsed key/value pairs as JSON. Stateless helper used by the
  Sidebar to autofill the form.
* ``POST /api/credentials`` — accepts the full credential set as JSON,
  persists it to ``.env`` (overwriting prior values), reloads ``.env`` into
  ``os.environ``, and resets the cached pipeline. The next ingest/query
  call rebuilds against the new values.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv, set_key
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from graphiti_rag.api.state import app_state

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_CREDS_FILE_BYTES = 64 * 1024

ENV_FILE = Path(os.environ.get("CREDS_ENV_FILE", ".env"))

REQUIRED_AURA_FILE_KEYS: tuple[str, ...] = (
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)

CRED_MAX = 512


class CredentialsPayload(BaseModel):
    neo4j_uri: str = Field("", max_length=CRED_MAX)
    neo4j_username: str = Field("", max_length=CRED_MAX)
    neo4j_password: str = Field("", max_length=CRED_MAX)
    neo4j_database: str = Field("", max_length=CRED_MAX)
    openai_api_key: str = Field("", max_length=CRED_MAX)
    aura_instanceid: str = Field("", max_length=CRED_MAX)
    aura_instancename: str = Field("", max_length=CRED_MAX)


@router.post("/credentials")
async def set_credentials(payload: CredentialsPayload) -> dict[str, str]:
    if not payload.openai_api_key.strip():
        raise HTTPException(status_code=400, detail="openai_api_key is required")

    uri = payload.neo4j_uri.strip()
    if (
        uri == ""
        and payload.aura_instanceid.strip() != ""
        and payload.aura_instancename.strip() != ""
    ):
        uri = f"neo4j+s://{payload.aura_instanceid.strip()}.databases.neo4j.io"
    if uri == "":
        raise HTTPException(
            status_code=400,
            detail="neo4j_uri is required (or supply both AURA instance ID and name)",
        )

    pairs: dict[str, str] = {
        "NEO4J_URI": uri,
        "NEO4J_USERNAME": payload.neo4j_username.strip(),
        "NEO4J_PASSWORD": payload.neo4j_password,
        "NEO4J_DATABASE": payload.neo4j_database.strip() or "neo4j",
        "OPENAI_API_KEY": payload.openai_api_key.strip(),
        "AURA_INSTANCEID": payload.aura_instanceid.strip(),
        "AURA_INSTANCENAME": payload.aura_instancename.strip(),
    }

    # `.env` may not exist on first run inside a fresh container.
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.touch(exist_ok=True)
    # Tighten file mode before writing so creds are never world-readable
    # even briefly. 0600 = owner read/write only.
    os.chmod(ENV_FILE, 0o600)

    for key, value in pairs.items():
        # quote_mode="auto" preserves passwords containing spaces, "#", or
        # newlines that would otherwise corrupt the .env file.
        set_key(str(ENV_FILE), key, value, quote_mode="auto")
    os.chmod(ENV_FILE, 0o600)

    # Push values into the running process so Graphiti's defaults (which
    # read OPENAI_API_KEY, etc.) see the new values immediately.
    load_dotenv(str(ENV_FILE), override=True)
    for key, value in pairs.items():
        os.environ[key] = value

    await app_state.reset()
    logger.info("Credentials persisted to %s; pipeline cleared", ENV_FILE)
    return {"status": "ok"}


@router.post("/credentials/parse")
async def parse_credentials(file: UploadFile = File(...)) -> dict[str, str]:
    raw = await file.read(MAX_CREDS_FILE_BYTES + 1)
    await file.close()
    if len(raw) > MAX_CREDS_FILE_BYTES:
        raise HTTPException(status_code=400, detail="Credentials file too large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Credentials file must be UTF-8 text")

    entries = _parse_dotenv_lines(text)
    missing = [k for k in REQUIRED_AURA_FILE_KEYS if k not in entries or entries[k] == ""]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Credentials file missing required keys: {', '.join(missing)}",
        )

    return {
        "neo4j_uri": entries["NEO4J_URI"],
        "neo4j_username": entries["NEO4J_USERNAME"],
        "neo4j_password": entries["NEO4J_PASSWORD"],
        "neo4j_database": entries["NEO4J_DATABASE"],
        "aura_instanceid": entries.get("AURA_INSTANCEID", ""),
        "aura_instancename": entries.get("AURA_INSTANCENAME", ""),
    }


def _parse_dotenv_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "" or line.startswith("#"):
            continue
        eq = line.find("=")
        if eq == -1:
            continue
        key = line[:eq].strip().upper()
        value = line[eq + 1 :].strip()
        if key == "":
            continue
        out[key] = value
    return out
