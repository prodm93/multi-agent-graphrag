"""FastAPI application factory."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from graphiti_rag.api.routes import credentials, ingest, query

logger = logging.getLogger(__name__)


def _user_facing_url() -> str:
    """The URL the user should open in a browser.

    Honours ``APP_PUBLIC_URL`` if set, otherwise picks the right default for
    the deployment shape: in Docker Compose nginx serves the SPA on port
    5173; in dev mode the Vite dev server uses the same port.
    """
    explicit = os.environ.get("APP_PUBLIC_URL", "").strip()
    if explicit:
        return explicit
    return "http://localhost:5173"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Use the uvicorn logger — when launched via `uvicorn`, only its own
    # loggers are guaranteed to have handlers attached. App loggers
    # (`graphiti_rag.*`) propagate to a root logger that uvicorn does not
    # configure, so INFO messages get silently dropped. Logging through
    # `uvicorn.error` (the standard channel for non-access uvicorn output)
    # ensures the banner actually reaches the terminal.
    uvicorn_logger = logging.getLogger("uvicorn.error")
    url = _user_facing_url()
    border = "─" * 60
    banner = (
        f"\n{border}\n"
        f"  Multi-agent GraphRAG is running.\n"
        f"  Open {url} in your browser.\n"
        f"{border}"
    )
    uvicorn_logger.info(banner)
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application with all routers and middleware attached."""
    app = FastAPI(
        title="multi-agent-graphrag",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:80",
            "http://localhost:5173",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest.router, prefix="/api")
    app.include_router(query.router, prefix="/api")
    app.include_router(credentials.router, prefix="/api")

    return app


app = create_app()
