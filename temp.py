"""
Experiments API routes (minimal functional v0).

Auto-creates schema/table:
- experiments.runs

You can replace this later with your full experiments design.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from api_gateway.deps import get_conn

router = APIRouter()


async def _ensure_experiments_schema(conn: AsyncConnection) -> None:
    # Safe to call repeatedly.
    await conn.execute(text("CREATE SCHEMA IF NOT EXISTS experiments;"))
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS experiments.runs (
          run_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          user_id      TEXT NOT NULL DEFAULT '',
          session_id   TEXT NOT NULL DEFAULT '',
          mode         TEXT NOT NULL DEFAULT 'chat',
          model_id     UUID NULL,
          status       TEXT NOT NULL DEFAULT 'created',
          params       JSONB NOT NULL DEFAULT '{}'::jsonb,
          summary      JSONB NOT NULL DEFAULT '{}'::jsonb,
          created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """))
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_experiments_runs_created_at
        ON experiments.runs (created_at DESC);
    """))


class ExperimentRun(BaseModel):
    run_id: str
    user_id: str = ""
    session_id: str = ""
    mode: str = "chat"
    model_id: Optional[str] = None
    status: str = "created"
    params: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class CreateRunRequest(BaseModel):
    user_id: str = ""
    session_id: str = ""
    mode: str = "chat"
    model_id: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


@router.get("/")
async def list_runs(
    limit: int = Query(50, ge=1, le=500),
    conn: AsyncConnection = Depends(get_conn),
) -> list[ExperimentRun]:
    await _ensure_experiments_schema(conn)

    rows = (await conn.execute(
        text("""
            SELECT run_id, user_id, session_id, mode, model_id, status, params, summary, created_at, updated_at
            FROM experiments.runs
            ORDER BY created_at DESC
            LIMIT :limit
        """),
        {"limit": limit},
    )).mappings().all()

    return [
        ExperimentRun(
            run_id=str(r["run_id"]),
            user_id=r["user_id"] or "",
            session_id=r["session_id"] or "",
            mode=r["mode"],
            model_id=str(r["model_id"]) if r["model_id"] else None,
            status=r["status"],
            params=dict(r["params"] or {}),
            summary=dict(r["summary"] or {}),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


@router.post("/")
async def create_run(
    req: CreateRunRequest,
    conn: AsyncConnection = Depends(get_conn),
) -> ExperimentRun:
    await _ensure_experiments_schema(conn)

    row = (await conn.execute(
        text("""
            INSERT INTO experiments.runs (user_id, session_id, mode, model_id, status, params)
            VALUES (:user_id, :session_id, :mode, :model_id, 'created', :params::jsonb)
            RETURNING run_id, user_id, session_id, mode, model_id, status, params, summary, created_at, updated_at
        """),
        {
            "user_id": req.user_id,
            "session_id": req.session_id,
            "mode": req.mode,
            "model_id": req.model_id,
            "params": json.dumps(req.params),
        },
    )).mappings().first()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to create run")

    await conn.commit()

    return ExperimentRun(
        run_id=str(row["run_id"]),
        user_id=row["user_id"] or "",
        session_id=row["session_id"] or "",
        mode=row["mode"],
        model_id=str(row["model_id"]) if row["model_id"] else None,
        status=row["status"],
        params=dict(row["params"] or {}),
        summary=dict(row["summary"] or {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("/{run_id}")
async def get_run(
    run_id: str,
    conn: AsyncConnection = Depends(get_conn),
) -> ExperimentRun:
    await _ensure_experiments_schema(conn)

    row = (await conn.execute(
        text("""
            SELECT run_id, user_id, session_id, mode, model_id, status, params, summary, created_at, updated_at
            FROM experiments.runs
            WHERE run_id = :run_id
            LIMIT 1
        """),
        {"run_id": run_id},
    )).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return ExperimentRun(
        run_id=str(row["run_id"]),
        user_id=row["user_id"] or "",
        session_id=row["session_id"] or "",
        mode=row["mode"],
        model_id=str(row["model_id"]) if row["model_id"] else None,
        status=row["status"],
        params=dict(row["params"] or {}),
        summary=dict(row["summary"] or {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
