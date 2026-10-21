"""Chat routes: conversations and messages."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User, Conversation, Message
from app.auth import get_current_user
from app.schemas.chat import (
    ChatRequest, ChatResponse, SourceItem,
    ConversationResponse, ConversationListResponse,
    MessageResponse, MessageListResponse,
)
from app.services.agent import get_agent_graph

router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = logging.getLogger("docmind")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    data: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Send a message and get an AI response with citations."""

    conversation = None
    conversation_history = []

    # Get or create conversation
    if data.conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == data.conversation_id,
                Conversation.user_id == user.id,
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Load conversation history
        msg_result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at)
        )
        messages = msg_result.scalars().all()
        conversation_history = [
            {"role": m.role, "content": m.content} for m in messages
        ]

    if not conversation:
        # Create new conversation with title from first message
        title = data.message[:100].strip()
        if len(data.message) > 100:
            title += "..."
        conversation = Conversation(
            user_id=user.id,
            title=title,
        )
        db.add(conversation)
        await db.flush()
        await db.refresh(conversation)

    # Store user message
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=data.message,
    )
    db.add(user_msg)
    await db.flush()

    # Run the LangGraph agent
    agent = get_agent_graph()

    initial_state = {
        "user_id": str(user.id),
        "conversation_id": str(conversation.id),
        "query": data.message,
        "rewritten_query": "",
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "document_ids": [str(d) for d in data.document_ids] if data.document_ids else None,
        "conversation_history": conversation_history,
        "db": db,
    }

    try:
        result = await agent.ainvoke(initial_state)
    except Exception as e:
        logger.exception(f"Agent error for conversation {conversation.id}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question. Please try again.",
        )

    answer = result.get("answer", "I couldn't generate a response. Please try again.")
    sources = result.get("sources", [])
    intent = result.get("intent", "rag")

    # Store assistant message
    sources_json = [
        {
            "document_id": s["document_id"],
            "filename": s["filename"],
            "page": s["page"],
            "snippet": s["snippet"],
            "score": s["score"],
        }
        for s in sources
    ] if sources else None

    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources=sources_json,
    )
    db.add(assistant_msg)
    await db.flush()

    return ChatResponse(
        conversation_id=conversation.id,
        answer=answer,
        sources=[SourceItem(**s) for s in sources] if sources else [],
        intent=intent,
    )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List all conversations for the current user."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
    )
    conversations = result.scalars().all()
    return ConversationListResponse(conversations=conversations)


@router.get("/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def get_conversation_messages(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get all messages in a conversation."""
    # Verify ownership
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = msg_result.scalars().all()

    return MessageListResponse(
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=[SourceItem(**s) for s in m.sources] if m.sources else None,
                created_at=m.created_at,
            )
            for m in messages
        ]
    )


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a conversation and its messages."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conv)
    logger.info(f"Conversation {conversation_id} deleted by user {user.id}")
