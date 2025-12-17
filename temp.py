from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # DB: asyncpg URL
    database_url: str = os.getenv(
        "MODELHUB_DB_URL",
        "postgresql://postgres:postgres@localhost:5432/agentic_suite",
    )

    # DB table (schema-qualified)
    table_fqdn: str = os.getenv("MODELHUB_TABLE", "modelhub.models")

    # Filesystem roots (used only when resolving absolute path for tools)
    official_root: str = os.getenv("MODELHUB_OFFICIAL_ROOT", "/data/modelhub/official/repo")
    overlay_root: str = os.getenv("MODELHUB_OVERLAY_ROOT", "/data/modelhub/overlays")

    # Server bind
    host: str = os.getenv("MODELHUB_MCP_HOST", "0.0.0.0")
    port: int = int(os.getenv("MODELHUB_MCP_PORT", "5310"))


def load_settings() -> Settings:
    return Settings()
