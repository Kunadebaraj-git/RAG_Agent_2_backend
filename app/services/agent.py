"""LangGraph agent for DocMind AI.

Graph: START -> classify_intent -> [rag_node | summarize_node | general_node] -> finalize -> END

State uses TypedDict with all fields needed across nodes.
user_id is ALWAYS from the authenticated session, never from LLM output.
"""

import logging
from typing import Optional, Literal
from uuid import UUID

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, Chunk, Message
from app.services.llm import get_llm_provider
from app.services.retrieval import retrieve_chunks
from app.prompts.templates import (
    RAG_SYSTEM, RAG_CONTEXT_TEMPLATE, RAG_DOCUMENT_CHUNK, RAG_USER_TEMPLATE,
    QUERY_REWRITE_SYSTEM, QUERY_REWRITE_USER,
    INTENT_CLASSIFY_SYSTEM, INTENT_CLASSIFY_USER,
    SUMMARIZE_MAP_SYSTEM, SUMMARIZE_MAP_USER,
    SUMMARIZE_REDUCE_SYSTEM, SUMMARIZE_REDUCE_USER,
)
from app.config import get_settings

logger = logging.getLogger("docmind")


# ============================================================
# State Definition
# ============================================================

class AgentState(TypedDict):
    user_id: str
    conversation_id: Optional[str]
    query: str
    rewritten_query: str
    intent: str
    retrieved_chunks: list[dict]
    answer: str
    sources: list[dict]
    document_ids: Optional[list[str]]
    conversation_history: list[dict]
    db: object  # AsyncSession, not serialized


# ============================================================
# Node Functions
# ============================================================

async def rewrite_query_node(state: AgentState) -> dict:
    """Rewrite follow-up questions into standalone queries."""
    history = state.get("conversation_history", [])

    if not history:
        return {"rewritten_query": state["query"]}

    llm = get_llm_provider()

    # Format recent history (last 6 messages)
    recent = history[-6:]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent
    )

    user_prompt = QUERY_REWRITE_USER.format(
        history=history_text,
        question=state["query"],
    )

    rewritten = await llm.generate(
        QUERY_REWRITE_SYSTEM,
        user_prompt,
        temperature=0.0,
        max_tokens=256,
    )

    logger.info(f"Query rewritten: '{state['query']}' -> '{rewritten.strip()}'")
    return {"rewritten_query": rewritten.strip() or state["query"]}


async def classify_intent_node(state: AgentState) -> dict:
    """Classify intent as rag, summarize, or general."""
    llm = get_llm_provider()

    query = state.get("rewritten_query") or state["query"]

    user_prompt = INTENT_CLASSIFY_USER.format(message=query)

    result = await llm.generate(
        INTENT_CLASSIFY_SYSTEM,
        user_prompt,
        temperature=0.0,
        max_tokens=10,
    )

    intent = result.strip().lower()

    # Validate - fall back to rag on garbage
    valid_intents = {"rag", "summarize", "general"}
    if intent not in valid_intents:
        logger.warning(f"Invalid intent '{intent}', falling back to 'rag'")
        intent = "rag"

    logger.info(f"Intent classified: {intent}")
    return {"intent": intent}


def route_by_intent(state: AgentState) -> str:
    """Route to the appropriate node based on intent."""
    intent = state.get("intent", "rag")
    if intent == "summarize":
        return "summarize_node"
    elif intent == "general":
        return "general_node"
    return "rag_node"


async def rag_node(state: AgentState) -> dict:
    """Retrieve relevant chunks and generate answer with citations."""
    db: AsyncSession = state["db"]
    settings = get_settings()

    query = state.get("rewritten_query") or state["query"]
    user_id = UUID(state["user_id"])

    # Parse document_ids if provided
    doc_ids = None
    if state.get("document_ids"):
        doc_ids = [UUID(did) for did in state["document_ids"]]

    # Retrieve chunks
    chunks = await retrieve_chunks(
        db=db,
        query=query,
        user_id=user_id,
        document_ids=doc_ids,
    )

    if not chunks:
        return {
            "answer": "I couldn't find enough information in your uploaded documents to answer that.",
            "sources": [],
            "retrieved_chunks": [],
        }

    # Build context
    context_parts = []
    for chunk in chunks:
        context_parts.append(RAG_DOCUMENT_CHUNK.format(
            filename=chunk["filename"],
            page=chunk["page"],
            text=chunk["text"],
        ))

    context = RAG_CONTEXT_TEMPLATE.format(context="\n".join(context_parts))

    # Build conversation context
    conversation_msgs = []
    for m in state.get("conversation_history", [])[-4:]:
        conversation_msgs.append(f"{m['role'].upper()}: {m['content']}")

    conversation_section = ""
    if conversation_msgs:
        conversation_section = "CONVERSATION HISTORY:\n" + "\n".join(conversation_msgs) + "\n\n"

    user_prompt = f"{conversation_section}{context}\n\n{RAG_USER_TEMPLATE.format(question=state['query'])}"

    # Generate
    llm = get_llm_provider()
    answer = await llm.generate(RAG_SYSTEM, user_prompt, temperature=0.2)

    # Format sources
    sources = [
        {
            "document_id": c["document_id"],
            "filename": c["filename"],
            "page": c["page"],
            "snippet": c["text"][:200],
            "score": c["score"],
        }
        for c in chunks
    ]

    return {
        "answer": answer.strip(),
        "sources": sources,
        "retrieved_chunks": chunks,
    }


async def summarize_node(state: AgentState) -> dict:
    """Summarize a specific document using map-reduce over chunks."""
    db: AsyncSession = state["db"]
    user_id = UUID(state["user_id"])
    query = state.get("rewritten_query") or state["query"]

    # Try to identify which document the user means
    result = await db.execute(
        select(Document).where(
            Document.user_id == user_id,
            Document.status == "indexed",
        )
    )
    documents = result.scalars().all()

    if not documents:
        return {
            "answer": "You don't have any indexed documents to summarize. Please upload a PDF first.",
            "sources": [],
            "retrieved_chunks": [],
        }

    # Try to match by filename mention in the query
    target_doc = None
    query_lower = query.lower()
    for doc in documents:
        if doc.filename.lower().replace(".pdf", "") in query_lower:
            target_doc = doc
            break

    # If no match, use the most recent document or the only one
    if not target_doc:
        if len(documents) == 1:
            target_doc = documents[0]
        else:
            # List available documents
            doc_list = "\n".join(f"- {d.filename}" for d in documents)
            return {
                "answer": f"Which document would you like me to summarize? Your documents:\n{doc_list}",
                "sources": [],
                "retrieved_chunks": [],
            }

    # Get all chunks for this document
    chunk_result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == target_doc.id)
        .order_by(Chunk.chunk_index)
    )
    chunks = chunk_result.scalars().all()

    if not chunks:
        return {
            "answer": f"No content found in {target_doc.filename}.",
            "sources": [],
            "retrieved_chunks": [],
        }

    llm = get_llm_provider()

    # Map phase: summarize groups of chunks (batch ~5 chunks per group to stay under context)
    group_size = 5
    section_summaries = []

    for i in range(0, len(chunks), group_size):
        group = chunks[i : i + group_size]
        pages = sorted(set(c.page_number for c in group))
        text = "\n\n".join(c.text for c in group)
        pages_str = ", ".join(str(p) for p in pages)

        user_prompt = SUMMARIZE_MAP_USER.format(
            filename=target_doc.filename,
            pages=pages_str,
            text=text,
        )

        summary = await llm.generate(
            SUMMARIZE_MAP_SYSTEM,
            user_prompt,
            temperature=0.2,
            max_tokens=512,
        )
        section_summaries.append(summary.strip())

    # Reduce phase: combine section summaries
    if len(section_summaries) == 1:
        final_summary = section_summaries[0]
    else:
        all_summaries = "\n\n---\n\n".join(
            f"Section {i+1}:\n{s}" for i, s in enumerate(section_summaries)
        )
        user_prompt = SUMMARIZE_REDUCE_USER.format(
            filename=target_doc.filename,
            summaries=all_summaries,
        )
        final_summary = await llm.generate(
            SUMMARIZE_REDUCE_SYSTEM,
            user_prompt,
            temperature=0.2,
            max_tokens=1024,
        )

    sources = [{
        "document_id": str(target_doc.id),
        "filename": target_doc.filename,
        "page": 1,
        "snippet": f"Summary of {target_doc.filename} ({target_doc.page_count} pages, {target_doc.chunk_count} chunks)",
        "score": 1.0,
    }]

    return {
        "answer": final_summary.strip(),
        "sources": sources,
        "retrieved_chunks": [],
    }


async def general_node(state: AgentState) -> dict:
    """Handle general/greeting messages without retrieval."""
    llm = get_llm_provider()

    system = (
        "You are DocMind AI, a friendly document Q&A assistant. "
        "The user is making small talk or a greeting. Respond briefly and helpfully. "
        "Mention that you can help them with questions about their uploaded PDF documents."
    )

    answer = await llm.generate(
        system,
        state["query"],
        temperature=0.5,
        max_tokens=256,
    )

    return {
        "answer": answer.strip(),
        "sources": [],
        "retrieved_chunks": [],
    }


# ============================================================
# Build the Graph
# ============================================================

def build_agent_graph() -> StateGraph:
    """Build and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("summarize_node", summarize_node)
    graph.add_node("general_node", general_node)

    # Set entry point
    graph.set_entry_point("rewrite_query")

    # Edges
    graph.add_edge("rewrite_query", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "rag_node": "rag_node",
            "summarize_node": "summarize_node",
            "general_node": "general_node",
        },
    )
    graph.add_edge("rag_node", END)
    graph.add_edge("summarize_node", END)
    graph.add_edge("general_node", END)

    return graph.compile()


# Singleton compiled graph
_agent_graph = None


def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph
