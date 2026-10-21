"""Document routes: upload, list, get, delete."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User, Document, Chunk
from app.auth import get_current_user
from app.schemas.document import DocumentResponse, DocumentListResponse
from app.services.ingestion import ingest_document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
logger = logging.getLogger("docmind")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_PAGES = 60
ALLOWED_MIME_TYPES = {"application/pdf"}


@router.post("", response_model=DocumentResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a PDF document for processing."""
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed")

    # Read file content (in memory, no disk)
    content = await file.read()

    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Validate it's actually a valid PDF
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(content))
        num_pages = len(reader.pages)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file")

    if num_pages == 0:
        raise HTTPException(status_code=400, detail="PDF has no pages")

    if num_pages > MAX_PAGES:
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {num_pages} pages. Maximum is {MAX_PAGES} pages"
        )

    # Check for extractable text
    has_text = False
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            has_text = True
            break

    if not has_text:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found. Scanned PDFs are not supported. "
                   "Please upload a PDF with selectable text."
        )

    # Create document record
    doc = Document(
        user_id=user.id,
        filename=file.filename,
        page_count=num_pages,
        status="processing",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    # Schedule background ingestion
    background_tasks.add_task(ingest_document, doc.id, user.id, content)

    logger.info(f"Document {doc.id} uploaded by user {user.id}: {file.filename} ({num_pages} pages)")
    return doc


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List all documents for the current user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user.id)
        .order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()
    return DocumentListResponse(documents=documents)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get a single document. Returns 404 for other users' documents."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a document and cascade-delete its chunks."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await db.delete(doc)
    logger.info(f"Document {document_id} deleted by user {user.id}")
