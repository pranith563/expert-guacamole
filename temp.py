from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import asyncpg


@dataclass
class DB:
    pool: asyncpg.Pool

    @classmethod
    async def connect(cls, database_url: str) -> "DB":
        pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=10)
        return cls(pool=pool)

    async def close(self) -> None:
        await self.pool.close()

    async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args: Any) -> Optional[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(sql, *args)
