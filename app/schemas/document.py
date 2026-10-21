"""Pydantic schemas for document endpoints."""

from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    page_count: int
    chunk_count: int
    status: str
    error: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
