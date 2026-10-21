"""Vector retrieval service using pgvector cosine similarity."""

import logging
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services.embedding import get_embedding_service

logger = logging.getLogger("docmind")


async def retrieve_chunks(
    db: AsyncSession,
    query: str,
    user_id: UUID,
    document_ids: list[UUID] | None = None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[dict]:
    """Retrieve top-k chunks by cosine similarity, always filtered by user_id.

    Returns list of {document_id, filename, page, text, score, chunk_index}.
    Results are ordered by similarity ranking (cosine similarity, highest first).
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K
    if similarity_threshold is None:
        similarity_threshold = settings.SIMILARITY_THRESHOLD

    embedding_service = get_embedding_service()

    # Embed the query
    query_embedding = await embedding_service.embed_single(query)

    # Build the query - ALWAYS filter by user_id for data isolation
    query_str = """
        SELECT
            c.id,
            c.document_id,
            d.filename,
            c.page_number,
            c.text,
            c.chunk_index,
            1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.user_id = :user_id
    """

    params = {
        "embedding": str(query_embedding),
        "user_id": str(user_id),
        "top_k": top_k,
        "threshold": similarity_threshold,
    }

    if document_ids:
        # Filter to specific documents
        doc_id_strs = [str(did) for did in document_ids]
        placeholders = ", ".join(f":doc_id_{i}" for i in range(len(doc_id_strs)))
        query_str += f" AND c.document_id IN ({placeholders})"
        for i, did in enumerate(doc_id_strs):
            params[f"doc_id_{i}"] = did

    query_str += """
        AND 1 - (c.embedding <=> CAST(:embedding AS vector)) >= :threshold
        ORDER BY c.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """

    result = await db.execute(text(query_str), params)
    rows = result.fetchall()

    return [
        {
            "chunk_id": str(row[0]),
            "document_id": str(row[1]),
            "filename": row[2],
            "page": row[3],
            "text": row[4],
            "chunk_index": row[5],
            "score": round(float(row[6]), 4),
        }
        for row in rows
    ]
