"""DocMind AI - FastAPI application entry point."""

import logging
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.errors import register_error_handlers
from app.routes import auth, health, documents, chat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("docmind")


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    # Startup validation
    missing = []
    if not settings.GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not settings.HF_API_TOKEN:
        missing.append("HF_API_TOKEN")
    if missing:
        logger.warning(
            f"Missing API keys: {', '.join(missing)}. "
            "Some features (embedding, LLM) will not work until configured."
        )

    application = FastAPI(
        title="DocMind AI",
        description="RAG-powered document Q&A with citations",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    if settings.FRONTEND_URL:
        for url in settings.FRONTEND_URL.split(","):
            cleaned = url.strip().rstrip("/")
            if cleaned:
                origins.append(cleaned)

    origins = list(dict.fromkeys([o for o in origins if o]))

    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=r"https://.*\.vercel\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handlers
    register_error_handlers(application)

    # Routes
    application.include_router(health.router)
    application.include_router(auth.router)
    application.include_router(documents.router)
    application.include_router(chat.router)

    return application


app = create_app()
