"""
Projects API routes.

Backed by modelhub.models in Postgres:
- Projects = split_part(file_path, '/', 1)
- Versions = split_part(file_path, '/', 2) per project

This is read-only for now (create/update not implemented).
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_gateway.deps import get_conn

router = APIRouter()

Source = Literal["official", "overlay"]


class Project(BaseModel):
    """Project model."""
    id: str
    name: str
    description: str = ""


class CreateProjectRequest(BaseModel):
    """Request to create a project."""
    name: str
    description: str = ""


class VersionsResponse(BaseModel):
    project: str
    source: Source
    owner_user_id: str
    versions: list[str]


@router.get("/")
async def list_projects(
    source: Source = Query("official"),
    owner_user_id: str = Query("", description="official uses empty string; overlay uses a user_id"),
    conn: AsyncConnection = Depends(get_conn),
) -> list[Project]:
    sql = text("""
        SELECT DISTINCT split_part(file_path, '/', 1) AS project
        FROM modelhub.models
        WHERE source = :source AND owner_user_id = :owner_user_id
        ORDER BY project
    """)
    rows = (await conn.execute(sql, {"source": source, "owner_user_id": owner_user_id})).all()
    return [Project(id=r.project, name=r.project, description="") for r in rows]


@router.post("/")
async def create_project(request: CreateProjectRequest) -> Project:
    """
    Creating projects in ModelHub implies creating directories + git/overlay workflow.
    We'll add this once overlay workflow is implemented.
    """
    raise HTTPException(
        status_code=501,
        detail="create_project is not implemented. Use overlay workflow / GitOps later.",
    )


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    source: Source = Query("official"),
    owner_user_id: str = Query(""),
    conn: AsyncConnection = Depends(get_conn),
) -> Project:
    sql = text("""
        SELECT 1
        FROM modelhub.models
        WHERE source = :source AND owner_user_id = :owner_user_id
          AND split_part(file_path, '/', 1) = :project
        LIMIT 1
    """)
    row = (await conn.execute(sql, {"source": source, "owner_user_id": owner_user_id, "project": project_id})).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return Project(id=project_id, name=project_id, description="")


@router.get("/{project_id}/versions", response_model=VersionsResponse)
async def list_versions(
    project_id: str,
    source: Source = Query("official"),
    owner_user_id: str = Query(""),
    conn: AsyncConnection = Depends(get_conn),
) -> VersionsResponse:
    sql = text("""
        SELECT DISTINCT split_part(file_path, '/', 2) AS version
        FROM modelhub.models
        WHERE source = :source AND owner_user_id = :owner_user_id
          AND split_part(file_path, '/', 1) = :project
        ORDER BY version
    """)
    rows = (await conn.execute(sql, {"source": source, "owner_user_id": owner_user_id, "project": project_id})).all()
    return VersionsResponse(
        project=project_id,
        source=source,
        owner_user_id=owner_user_id,
        versions=[r.version for r in rows if r.version],
    )
