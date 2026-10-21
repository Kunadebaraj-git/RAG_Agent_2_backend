"""PDF ingestion pipeline: extract text, chunk, embed, store."""

import logging
import io
from uuid import UUID

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import async_session
from app.models import Document, Chunk
from app.services.embedding import get_embedding_service

logger = logging.getLogger("docmind")


def extract_pages_from_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text per page from PDF bytes. Returns list of (page_number, text)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i + 1, text))  # 1-indexed page numbers
    return pages


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Collapse multiple whitespace/newlines
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Split pages into chunks preserving page numbers.

    Returns list of {"chunk_index": int, "page_number": int, "text": str}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_index = 0
    for page_num, page_text in pages:
        cleaned = clean_text(page_text)
        if not cleaned:
            continue
        page_chunks = splitter.split_text(cleaned)
        for chunk_text in page_chunks:
            chunks.append({
                "chunk_index": chunk_index,
                "page_number": page_num,
                "text": chunk_text,
            })
            chunk_index += 1

    return chunks


async def ingest_document(document_id: UUID, user_id: UUID, pdf_bytes: bytes):
    """Full ingestion pipeline: extract -> chunk -> embed -> store.

    Updates document status to 'indexed' or 'failed'.
    """
    settings = get_settings()
    embedding_service = get_embedding_service()

    try:
        # 1. Extract text per page
        pages = extract_pages_from_pdf(pdf_bytes)

        if not pages:
            await _update_document_status(
                document_id, "failed",
                error="No extractable text found. Scanned PDFs are not supported.",
                page_count=0,
            )
            return

        page_count = max(p[0] for p in pages)

        # 2. Chunk with page numbers
        chunks = chunk_pages(pages, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        if not chunks:
            await _update_document_status(
                document_id, "failed",
                error="No text chunks produced from document.",
                page_count=page_count,
            )
            return

        logger.info(f"Document {document_id}: {page_count} pages, {len(chunks)} chunks")

        # 3. Embed in batches
        texts = [c["text"] for c in chunks]
        embeddings = await embedding_service.embed_batch(texts)

        # 4. Bulk insert chunks into DB
        async with async_session() as session:
            chunk_objects = []
            for chunk_data, embedding in zip(chunks, embeddings):
                chunk_objects.append(Chunk(
                    document_id=document_id,
                    user_id=user_id,
                    chunk_index=chunk_data["chunk_index"],
                    page_number=chunk_data["page_number"],
                    text=chunk_data["text"],
                    embedding=embedding,
                ))

            session.add_all(chunk_objects)

            # Update document status
            await session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(
                    status="indexed",
                    page_count=page_count,
                    chunk_count=len(chunks),
                    error=None,
                )
            )
            await session.commit()

        logger.info(f"Document {document_id}: ingestion complete")

    except Exception as e:
        logger.exception(f"Document {document_id}: ingestion failed")
        await _update_document_status(
            document_id, "failed",
            error=f"Ingestion failed: {str(e)[:500]}",
        )


async def _update_document_status(
    document_id: UUID,
    status: str,
    error: str = None,
    page_count: int = None,
):
    """Update document status in the database."""
    async with async_session() as session:
        values = {"status": status, "error": error}
        if page_count is not None:
            values["page_count"] = page_count
        await session.execute(
            update(Document).where(Document.id == document_id).values(**values)
        )
        await session.commit()
