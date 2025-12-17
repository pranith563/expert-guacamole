from __future__ import annotations

from typing import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine


async def get_conn(request: Request) -> AsyncIterator[AsyncConnection]:
    """Get an AsyncConnection from the app engine (one per request)."""
    engine: AsyncEngine = request.app.state.db_engine
    async with engine.connect() as conn:
        yield conn
