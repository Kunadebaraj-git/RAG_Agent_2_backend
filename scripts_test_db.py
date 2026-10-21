import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.config import get_settings
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text


async def inspect_db():
    settings = get_settings()
    url = settings.DATABASE_URL
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)

    print("Connecting to DB...")
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        res = await conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
        ))
        tables = [r[0] for r in res.fetchall()]
        print("Existing tables in public schema:", tables)

        res2 = await conn.execute(text(
            "SELECT version_num FROM alembic_version;"
        ))
        alembic_ver = [r[0] for r in res2.fetchall()] if "alembic_version" in tables else None
        print("Alembic version in DB:", alembic_ver)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(inspect_db())
