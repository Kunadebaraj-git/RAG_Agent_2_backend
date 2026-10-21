"""Pydantic schemas for chat endpoints."""

from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional


class SourceItem(BaseModel):
    document_id: UUID
    filename: str
    page: int
    snippet: str
    score: float


class ChatRequest(BaseModel):
    conversation_id: Optional[UUID] = None
    message: str = Field(min_length=1, max_length=5000)
    document_ids: Optional[list[UUID]] = None


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    sources: list[SourceItem]
    intent: str


class ConversationResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    sources: Optional[list[SourceItem]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]
