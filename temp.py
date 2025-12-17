from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, Optional

from starlette.applications import Starlette
from starlette.routing import Mount

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from .db import DB
from .settings import Settings, load_settings

Source = Literal["official", "overlay"]


@dataclass
class AppCtx:
    settings: Settings
    db: DB


def _norm_rel(p: str) -> str:
    # keep relative
    return (p or "").strip().strip("/")


def _overlay_repo_root(settings: Settings, owner_user_id: Optional[str]) -> str:
    # overlays/<user>/repo/<relative file_path>
    user = owner_user_id or "default"
    return os.path.join(settings.overlay_root, user, "repo")


def _resolve_abs_path(settings: Settings, file_path: str, source: Source, owner_user_id: Optional[str]) -> str:
    rel = _norm_rel(file_path)
    if source == "official":
        return os.path.join(settings.official_root, rel)
    return os.path.join(_overlay_repo_root(settings, owner_user_id), rel)


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[AppCtx]:
    settings = load_settings()
    db = await DB.connect(settings.database_url)
    try:
        yield AppCtx(settings=settings, db=db)
    finally:
        await db.close()


# Streamable HTTP MCP server
mcp = FastMCP(
    "ModelHubMCP",
    lifespan=lifespan,
    stateless_http=True,
    json_response=True,
)

# Mount at /mcp and make streamable path exactly "/"
mcp.settings.streamable_http_path = "/"


def _tbl(ctx: AppCtx) -> str:
    # Table name is config-controlled; do not accept from user.
    return ctx.settings.table_fqdn


@mcp.tool()
async def modelhub_list_projects(
    source: Source = "official",
    owner_user_id: str = "",
    ctx: Context[ServerSession, AppCtx] = None,
) -> dict[str, Any]:
    """List projects (top-level folders)."""
    app = ctx.request_context.lifespan_context
    t = _tbl(app)

    sql = f"""
        SELECT DISTINCT split_part(file_path, '/', 1) AS project
        FROM {t}
        WHERE source = $1 AND owner_user_id = $2
        ORDER BY project
    """
    rows = await app.db.fetch(sql, source, owner_user_id)
    return {"source": source, "owner_user_id": owner_user_id, "projects": [r["project"] for r in rows]}


@mcp.tool()
async def modelhub_list_versions(
    project: str,
    source: Source = "official",
    owner_user_id: str = "",
    ctx: Context[ServerSession, AppCtx] = None,
) -> dict[str, Any]:
    """List versions under a project (2nd path segment)."""
    app = ctx.request_context.lifespan_context
    t = _tbl(app)

    sql = f"""
        SELECT DISTINCT split_part(file_path, '/', 2) AS version
        FROM {t}
        WHERE source = $1 AND owner_user_id = $2
          AND split_part(file_path, '/', 1) = $3
        ORDER BY version
    """
    rows = await app.db.fetch(sql, source, owner_user_id, project)
    versions = [r["version"] for r in rows if r["version"]]
    return {"project": project, "source": source, "owner_user_id": owner_user_id, "versions": versions}


@mcp.tool()
async def modelhub_browse(
    prefix: str = "",
    source: Source = "official",
    owner_user_id: str = "",
    ctx: Context[ServerSession, AppCtx] = None,
) -> dict[str, Any]:
    """
    Browse a path prefix and return immediate child folders + files.
    Returns file metadata for files at that level.
    """
    app = ctx.request_context.lifespan_context
    t = _tbl(app)

    prefix_norm = _norm_rel(prefix)

    # Root browse: return top-level folders
    if prefix_norm == "":
        sql = f"""
            SELECT DISTINCT split_part(ltrim(file_path, '/'), '/', 1) AS name
            FROM {t}
            WHERE source = $1 AND owner_user_id = $2
              AND split_part(ltrim(file_path, '/'), '/', 1) <> ''
            ORDER BY name
        """
        rows = await app.db.fetch(sql, source, owner_user_id)
        return {"prefix": "", "folders": [r["name"] for r in rows], "files": []}

    like_prefix = f"{prefix_norm}/%"

    sql = f"""
        WITH base AS (
          SELECT
            model_id,
            file_path,
            artifact_name,
            file_type,
            framework,
            size_bytes,
            sha256,
            substring(file_path from length($3) + 2) AS rest
          FROM {t}
          WHERE source = $1
            AND owner_user_id = $2
            AND file_path LIKE $4
        ),
        folders AS (
          SELECT DISTINCT split_part(rest, '/', 1) AS name
          FROM base
          WHERE rest LIKE '%/%'
        ),
        files AS (
          SELECT
            model_id,
            rest AS name,
            file_path,
            artifact_name,
            file_type,
            framework,
            size_bytes,
            sha256
          FROM base
          WHERE rest NOT LIKE '%/%'
        )
        SELECT 'folder' AS kind,
               name,
               NULL::uuid AS model_id,
               NULL::text AS file_path,
               NULL::text AS artifact_name,
               NULL::text AS file_type,
               NULL::text AS framework,
               NULL::bigint AS size_bytes,
               NULL::text AS sha256
        FROM folders

        UNION ALL

        SELECT 'file' AS kind,
               name,
               model_id,
               file_path,
               artifact_name,
               file_type,
               framework,
               size_bytes,
               sha256
        FROM files

        ORDER BY kind, name
    """
    rows = await app.db.fetch(sql, source, owner_user_id, prefix_norm, like_prefix)

    folders: list[str] = []
    files: list[dict[str, Any]] = []

    for r in rows:
        if r["kind"] == "folder":
            folders.append(r["name"])
        else:
            files.append({
                "model_id": str(r["model_id"]),
                "name": r["name"],
                "file_path": r["file_path"],
                "artifact_name": r["artifact_name"],
                "file_type": r["file_type"],
                "framework": r["framework"],
                "size_bytes": int(r["size_bytes"] or 0),
                "sha256": r["sha256"],
                "abs_path": _resolve_abs_path(app.settings, r["file_path"], source, owner_user_id),
            })

    return {"prefix": prefix_norm, "folders": folders, "files": files}


@mcp.tool()
async def modelhub_get_model(
    model_id: str,
    source: Source = "official",
    owner_user_id: str = "",
    ctx: Context[ServerSession, AppCtx] = None,
) -> dict[str, Any]:
    """Fetch a model artifact by model_id."""
    app = ctx.request_context.lifespan_context
    t = _tbl(app)

    sql = f"""
        SELECT model_id, file_path, artifact_name, file_type, framework, size_bytes, sha256
        FROM {t}
        WHERE source = $1 AND owner_user_id = $2
          AND model_id = $3::uuid
        LIMIT 1
    """
    row = await app.db.fetchrow(sql, source, owner_user_id, model_id)
    if not row:
        return {"found": False, "model_id": model_id}

    return {
        "found": True,
        "model_id": str(row["model_id"]),
        "file_path": row["file_path"],
        "artifact_name": row["artifact_name"],
        "file_type": row["file_type"],
        "framework": row["framework"],
        "size_bytes": int(row["size_bytes"] or 0),
        "sha256": row["sha256"],
        "abs_path": _resolve_abs_path(app.settings, row["file_path"], source, owner_user_id),
    }


@mcp.tool()
async def modelhub_get_by_path(
    file_path: str,
    source: Source = "official",
    owner_user_id: str = "",
    ctx: Context[ServerSession, AppCtx] = None,
) -> dict[str, Any]:
    """Fetch a model artifact by relative file_path."""
    app = ctx.request_context.lifespan_context
    t = _tbl(app)

    fp = _norm_rel(file_path)

    sql = f"""
        SELECT model_id, file_path, artifact_name, file_type, framework, size_bytes, sha256
        FROM {t}
        WHERE source = $1 AND owner_user_id = $2
          AND file_path = $3
        LIMIT 1
    """
    row = await app.db.fetchrow(sql, source, owner_user_id, fp)
    if not row:
        return {"found": False, "file_path": fp}

    return {
        "found": True,
        "model_id": str(row["model_id"]),
        "file_path": row["file_path"],
        "artifact_name": row["artifact_name"],
        "file_type": row["file_type"],
        "framework": row["framework"],
        "size_bytes": int(row["size_bytes"] or 0),
        "sha256": row["sha256"],
        "abs_path": _resolve_abs_path(app.settings, row["file_path"], source, owner_user_id),
    }


@mcp.tool()
async def modelhub_search(
    query: str,
    source: Source = "official",
    owner_user_id: str = "",
    limit: int = 20,
    ctx: Context[ServerSession, AppCtx] = None,
) -> dict[str, Any]:
    """Simple ILIKE search over artifact_name and file_path."""
    app = ctx.request_context.lifespan_context
    t = _tbl(app)

    q = f"%{(query or '').strip()}%"
    sql = f"""
        SELECT model_id, file_path, artifact_name, file_type, framework, size_bytes, sha256
        FROM {t}
        WHERE source = $1 AND owner_user_id = $2
          AND (
            artifact_name ILIKE $3 OR
            file_path ILIKE $3
          )
        ORDER BY file_path
        LIMIT $4
    """
    rows = await app.db.fetch(sql, source, owner_user_id, q, limit)

    return {
        "query": query,
        "results": [
            {
                "model_id": str(r["model_id"]),
                "file_path": r["file_path"],
                "artifact_name": r["artifact_name"],
                "file_type": r["file_type"],
                "framework": r["framework"],
                "size_bytes": int(r["size_bytes"] or 0),
                "sha256": r["sha256"],
                "abs_path": _resolve_abs_path(app.settings, r["file_path"], source, owner_user_id),
            }
            for r in rows
        ],
    }


# ---- ASGI app mount at /mcp ----
@asynccontextmanager
async def starlette_lifespan(_: Starlette):
    # ensures mcp session manager is running
    async with mcp.session_manager.run():
        yield


app = Starlette(
    routes=[Mount("/mcp", mcp.streamable_http_app())],
    lifespan=starlette_lifespan,
)

export MODELHUB_DB_URL="postgresql://postgres:postgres@localhost:5432/agentic_suite"
export MODELHUB_TABLE="modelhub.models"
export MODELHUB_OFFICIAL_ROOT="/data/modelhub/official/repo"
export MODELHUB_MCP_PORT=5310
