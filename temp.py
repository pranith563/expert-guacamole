"""
Models API routes (ModelHub artifacts).

Backed by modelhub.models in Postgres.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_gateway.deps import get_conn

router = APIRouter()

Source = Literal["official", "overlay"]


class Model(BaseModel):
    """Model metadata (one artifact row)."""
    id: str
    name: str
    framework: str
    path: str


class FileItem(BaseModel):
    model_id: str
    name: str
    file_path: str
    file_type: str
    framework: Optional[str] = None
    size_bytes: int
    sha256: str


class BrowseResponse(BaseModel):
    prefix: str
    folders: list[str]
    files: list[FileItem]


@router.get("/")
async def list_models(
    # Common filters
    source: Source = Query("official"),
    owner_user_id: str = Query("", description="official uses empty string; overlay uses a user_id"),
    project: Optional[str] = Query(None),
    version: Optional[str] = Query(None),
    prefix: Optional[str] = Query(None, description="Path prefix like 'HC/P2Q_HC'"),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    conn: AsyncConnection = Depends(get_conn),
) -> list[Model]:
    """
    Returns artifact rows as a flat list. UI can use /browse for tree navigation.
    """
    where = ["source = :source", "owner_user_id = :owner_user_id"]
    params: dict[str, Any] = {
        "source": source,
        "owner_user_id": owner_user_id,
        "limit": limit,
        "offset": offset,
    }

    if project:
        where.append("split_part(file_path, '/', 1) = :project")
        params["project"] = project
    if version:
        where.append("split_part(file_path, '/', 2) = :version")
        params["version"] = version
    if prefix:
        p = prefix.strip().strip("/")
        where.append("file_path LIKE :prefix_like")
        params["prefix_like"] = f"{p}/%"

    sql = text(f"""
        SELECT model_id, artifact_name, framework, file_path
        FROM modelhub.models
        WHERE {' AND '.join(where)}
        ORDER BY file_path
        LIMIT :limit OFFSET :offset
    """)

    rows = (await conn.execute(sql, params)).all()
    return [
        Model(
            id=str(r.model_id),
            name=r.artifact_name,
            framework=r.framework or "",
            path=r.file_path,
        )
        for r in rows
    ]


@router.get("/browse", response_model=BrowseResponse)
async def browse_prefix(
    prefix: str = Query(..., description="Directory-like prefix e.g. 'HC/P2Q_HC'"),
    source: Source = Query("official"),
    owner_user_id: str = Query(""),
    conn: AsyncConnection = Depends(get_conn),
) -> BrowseResponse:
    """
    Returns immediate child folders + files under the given prefix.
    Mirrors filesystem structure via modelhub.models.file_path.
    """
    prefix_norm = prefix.strip().strip("/")
    like_prefix = f"{prefix_norm}/%"

    sql = text("""
        WITH base AS (
          SELECT
            model_id,
            file_path,
            file_type,
            framework,
            size_bytes,
            sha256,
            substring(file_path from length(:prefix) + 2) AS rest
          FROM modelhub.models
          WHERE source = :source
            AND owner_user_id = :owner_user_id
            AND file_path LIKE :like_prefix
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
               file_type,
               framework,
               size_bytes,
               sha256
        FROM files
        ORDER BY kind, name
    """)

    rows = (await conn.execute(sql, {
        "source": source,
        "owner_user_id": owner_user_id,
        "prefix": prefix_norm,
        "like_prefix": like_prefix,
    })).mappings().all()

    folders: list[str] = []
    files: list[FileItem] = []

    for r in rows:
        if r["kind"] == "folder":
            folders.append(r["name"])
        else:
            files.append(FileItem(
                model_id=str(r["model_id"]),
                name=r["name"],
                file_path=r["file_path"],
                file_type=r["file_type"],
                framework=r["framework"],
                size_bytes=int(r["size_bytes"]),
                sha256=r["sha256"],
            ))

    return BrowseResponse(prefix=prefix_norm, folders=folders, files=files)


@router.post("/upload")
async def upload_model() -> Model:
    """
    Upload will be implemented with overlays (multipart upload -> /data/modelhub/tmp -> atomic move into overlays).
    """
    raise HTTPException(
        status_code=501,
        detail="upload_model is not implemented yet. Use overlay workflow later.",
    )


@router.get("/{model_id}")
async def get_model(
    model_id: str,
    conn: AsyncConnection = Depends(get_conn),
) -> Model:
    sql = text("""
        SELECT model_id, artifact_name, framework, file_path
        FROM modelhub.models
        WHERE model_id = :model_id
        LIMIT 1
    """)
    row = (await conn.execute(sql, {"model_id": model_id})).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return Model(
        id=str(row.model_id),
        name=row.artifact_name,
        framework=row.framework or "",
        path=row.file_path,
    )
