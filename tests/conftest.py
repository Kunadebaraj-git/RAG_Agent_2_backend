"""Shared test fixtures and configuration."""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Set test environment variables BEFORE any app imports
os.environ.update({
    "DATABASE_URL": "postgresql+psycopg://test:test@localhost:5432/testdb",
    "SECRET_KEY": "test-secret-key-for-testing-only",
    "GROQ_API_KEY": "test-groq-key",
    "HF_API_TOKEN": "test-hf-token",
    "FRONTEND_URL": "http://localhost:5173",
})

from app.auth import hash_password, verify_password, create_access_token, decode_token
from app.services.ingestion import extract_pages_from_pdf, chunk_pages, clean_text
from app.prompts.templates import RAG_SYSTEM


@pytest.fixture
def sample_pdf_bytes():
    """Create a minimal valid PDF in memory."""
    from pypdf import PdfWriter
    import io

    writer = PdfWriter()
    # Add a page with text
    from pypdf._page import PageObject
    page = PageObject.create_blank_page(width=612, height=792)
    writer.add_page(page)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def empty_pdf_bytes():
    """A valid PDF with no extractable text (simulates scanned)."""
    from pypdf import PdfWriter
    import io

    writer = PdfWriter()
    from pypdf._page import PageObject
    page = PageObject.create_blank_page(width=612, height=792)
    writer.add_page(page)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def corrupt_pdf_bytes():
    """Not a valid PDF at all."""
    return b"This is not a PDF file at all. Just random bytes."


@pytest.fixture
def mock_embedding_service():
    """Mock the embedding service to avoid HF API calls."""
    mock = AsyncMock()
    mock.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    mock.embed_single = AsyncMock(return_value=[0.1] * 384)
    return mock


@pytest.fixture
def mock_llm_provider():
    """Mock the LLM provider to avoid Groq API calls."""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value="Test response")
    mock.generate_with_history = AsyncMock(return_value="Test response")
    return mock
