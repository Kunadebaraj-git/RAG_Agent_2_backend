import sys
import asyncio

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import get_settings

_engine = None
_async_session = None


def get_engine():
    global _engine
    if _engine is None:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        settings = get_settings()
        url = settings.DATABASE_URL
        if "channel_binding=" in url:
            import re
            url = re.sub(r"[?&]channel_binding=[^&]+", "", url)
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        _engine = create_async_engine(
            url,
            echo=False,
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,
        )
    return _engine


def get_sessionmaker():
    global _async_session
    if _async_session is None:
        _async_session = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _async_session


class SessionMakerProxy:
    def __call__(self):
        return get_sessionmaker()()

async_session = SessionMakerProxy()


async def get_db() -> AsyncSession:
    """Dependency that yields an async DB session."""
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

