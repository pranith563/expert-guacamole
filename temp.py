"""
Chat API routes - interface to orchestrator (HTTP proxy).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api_gateway.deps import get_orchestrator_client

router = APIRouter()

Mode = Literal["chat", "convert", "profile", "patch_search"]


class UIContext(BaseModel):
    selected_project_id: Optional[str] = None
    selected_model_id: Optional[str] = None
    selected_artifact_id: Optional[str] = None


class ChatMessage(BaseModel):
    """Chat message (UI-facing)."""
    role: Literal["user", "assistant", "system"] = "assistant"
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """
    Chat request from UI.
    """
    user_id: str = "ui_user"
    session_id: str = Field(default_factory=lambda: f"sess_{int(time.time() * 1000)}")

    message: str
    mode: Mode = "chat"
    ui_context: UIContext = Field(default_factory=UIContext)

    # extra knobs per mode (convert/profile/patch_search)
    mode_params: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """
    Chat response to UI.
    """
    session_id: str
    mode: Mode
    messages: list[ChatMessage]
    status: Literal["completed", "failed"]


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    orchestrator: httpx.AsyncClient = Depends(get_orchestrator_client),
) -> ChatResponse:
    """
    Proxies a chat request to the orchestrator service (/api/agent/interact).
    """
    payload = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "mode": request.mode,
        "ui_context": request.ui_context.model_dump(),
        "message": {
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat(),
        },
        "mode_params": request.mode_params,
        # history intentionally omitted (you disabled in frontend; orchestrator uses thread_id)
    }

    try:
        resp = await orchestrator.post("/api/agent/interact", json=payload)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach orchestrator service: {e}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Orchestrator error: {resp.text}",
        )

    data = resp.json()

    # Orchestrator response expected:
    # { session_id, mode, ui_context, messages: [...], agent_state, mode_params, ... }
    messages_out: list[ChatMessage] = []
    for m in data.get("messages", []):
        role = m.get("role", "assistant")
        if role not in ("user", "assistant", "system"):
            role = "assistant"
        messages_out.append(
            ChatMessage(
                role=role,  # type: ignore
                content=m.get("content", ""),
                timestamp=None,
            )
        )

    return ChatResponse(
        session_id=data.get("session_id", request.session_id),
        mode=data.get("mode", request.mode),
        messages=messages_out,
        status="completed",
    )


@router.get("/modes")
async def list_modes() -> list[dict[str, str]]:
    """List available chat modes."""
    return [
        {"id": "chat", "name": "Chat", "description": "General conversation"},
        {"id": "convert", "name": "Convert", "description": "Model conversion"},
        {"id": "profile", "name": "Profile", "description": "Model profiling"},
        {"id": "patch_search", "name": "Patch Search", "description": "Search for solutions"},
    ]
