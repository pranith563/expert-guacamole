"""
FastAPI application entrypoint.

Provides HTTP API for the UI to interact with:
- ModelHub (projects/versions/models browsing from Postgres index)
- Experiments (minimal run tracking; auto-creates schema/table)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from agentic_logging import get_logger, setup_logging
from api_gateway.routes import projects, models, experiments, chat

# Setup logging
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger("api-gateway")


def _make_engine() -> AsyncEngine:
    # Uses asyncpg driver.
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_suite",
    )
    return create_async_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    logger.info("Starting API Gateway")

    app.state.db_engine = _make_engine()

    # Optional: sanity check connection at startup (fail fast)
    try:
        async with app.state.db_engine.connect() as conn:
            await conn.execute("SELECT 1")
        logger.info("Postgres connection OK")
    except Exception:
        logger.exception("Postgres connection failed at startup")
        raise

    yield

    logger.info("Shutting down API Gateway")
    await app.state.db_engine.dispose()


app = FastAPI(
    title="Agentic DL Workflow Suite API",
    description="API for model conversion and deployment workflows",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


def run() -> None:
    """Run the API server."""
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
