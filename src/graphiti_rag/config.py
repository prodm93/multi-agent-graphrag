"""Application configuration.

Uses `pydantic-settings` so the same `Config` class supports two modes:

* **Dev mode** — instantiate with no arguments; values are read from `.env`.
* **App mode** — instantiate per-request with credentials supplied by the
  client; nothing is read from disk and nothing is cached server-side.

If `AURA_INSTANCEID` and `AURA_INSTANCENAME` are both present in the
environment (and `neo4j_uri` is not otherwise supplied), the URI is
derived as ``neo4j+s://<AURA_INSTANCEID>.databases.neo4j.io`` so dev mode
can target an AuraDB instance without restating the URI.
"""
from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment or direct injection.

    Credentials fields must never be logged. Any logging that includes
    request context must redact ``openai_api_key`` and ``neo4j_password``.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Neo4j (from Aura .txt download)
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    neo4j_database: str = ""

    # AuraDB instance metadata — when both are set and neo4j_uri is empty,
    # the URI is derived from the instance ID at validation time.
    aura_instanceid: str = ""
    aura_instancename: str = ""

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Graphiti
    graphiti_namespace: str = "default"

    # LangSmith (developer-only observability). All four fields are managed
    # by the developer's `.env`; end users never see them. The startup
    # probe in api/app.py forces ``langsmith_tracing`` off if the key
    # fails to authenticate, regardless of what `.env` says.
    langsmith_api_key: str = ""
    langsmith_project: str = ""
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    @model_validator(mode="after")
    def _derive_aura_uri(self) -> "Config":
        if (
            self.neo4j_uri == ""
            and self.aura_instanceid != ""
            and self.aura_instancename != ""
        ):
            self.neo4j_uri = (
                f"neo4j+s://{self.aura_instanceid}.databases.neo4j.io"
            )
        return self
