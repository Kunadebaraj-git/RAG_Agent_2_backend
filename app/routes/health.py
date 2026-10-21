"""Health check endpoint."""

import logging
from fastapi import APIRouter
from sqlalchemy import text

from app.database import get_sessionmaker
from app.config import get_settings

router = APIRouter(tags=["health"])
logger = logging.getLogger("docmind")


@router.get("/health")
async def health():
    settings = get_settings()
    result = {
        "status": "ok",
        "database": False,
        "pgvector": False,
        "groq_configured": bool(settings.GROQ_API_KEY),
        "hf_configured": bool(settings.HF_API_TOKEN),
    }

    try:
        session_factory = get_sessionmaker()
        async with session_factory() as session:
            # Check DB connection
            await session.execute(text("SELECT 1"))
            result["database"] = True

            # Check pgvector extension
            row = await session.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
            result["pgvector"] = row.scalar_one_or_none() is not None
    except Exception as e:
        print(f"HEALTH CHECK ERROR: {e}", flush=True)
        logger.exception(f"Health check DB error: {e}")

    if not all([result["database"], result["pgvector"]]):
        result["status"] = "degraded"

    return result
