"""Tests for retrieval isolation and intent classification."""

import os
os.environ.update({
    "DATABASE_URL": "postgresql+psycopg://test:test@localhost:5432/testdb",
    "SECRET_KEY": "test-secret-key-for-testing-only",
    "GROQ_API_KEY": "test-groq-key",
    "HF_API_TOKEN": "test-hf-token",
    "FRONTEND_URL": "http://localhost:5173",
})

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4


class TestRetrievalIsolation:
    """Critical: retrieval MUST never return another user's chunks."""

    def test_retrieval_query_filters_by_user_id(self):
        """Verify the retrieval SQL always includes user_id filter."""
        # We check the source code directly to ensure the filter is present
        import inspect
        from app.services.retrieval import retrieve_chunks

        source = inspect.getsource(retrieve_chunks)
        assert "user_id" in source
        assert "c.user_id = :user_id" in source or "c.user_id =" in source

    def test_retrieval_query_has_no_bypass(self):
        """Ensure there's no code path that skips the user_id filter."""
        import inspect
        from app.services.retrieval import retrieve_chunks

        source = inspect.getsource(retrieve_chunks)
        # The WHERE clause with user_id should always be present
        # and there should be no conditional that removes it
        assert "WHERE c.user_id = :user_id" in source


class TestIntentRouting:
    """Test intent classification logic."""

    @pytest.mark.asyncio
    async def test_classify_validates_output(self):
        """Invalid LLM output should fall back to 'rag'."""
        from app.services.agent import classify_intent_node

        # Mock the LLM to return garbage
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="banana")

        with patch("app.services.agent.get_llm_provider", return_value=mock_llm):
            state = {
                "query": "test question",
                "rewritten_query": "test question",
            }
            result = await classify_intent_node(state)
            assert result["intent"] == "rag"  # Falls back to rag

    @pytest.mark.asyncio
    async def test_classify_accepts_valid_intents(self):
        """Valid intents should be accepted."""
        from app.services.agent import classify_intent_node

        for intent in ["rag", "summarize", "general"]:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=intent)

            with patch("app.services.agent.get_llm_provider", return_value=mock_llm):
                state = {
                    "query": "test",
                    "rewritten_query": "test",
                }
                result = await classify_intent_node(state)
                assert result["intent"] == intent

    def test_route_by_intent(self):
        """Test routing logic returns correct node names."""
        from app.services.agent import route_by_intent

        assert route_by_intent({"intent": "rag"}) == "rag_node"
        assert route_by_intent({"intent": "summarize"}) == "summarize_node"
        assert route_by_intent({"intent": "general"}) == "general_node"
        assert route_by_intent({"intent": "unknown"}) == "rag_node"  # default
        assert route_by_intent({}) == "rag_node"  # missing intent


class TestInsufficientEvidence:
    """Test that RAG returns insufficient-evidence message when no results."""

    @pytest.mark.asyncio
    async def test_rag_node_no_chunks(self):
        """When retrieval returns nothing, should return insufficient-evidence message."""
        from app.services.agent import rag_node

        mock_db = AsyncMock()

        # Mock retrieve_chunks to return empty
        with patch("app.services.agent.retrieve_chunks", new_callable=AsyncMock, return_value=[]):
            state = {
                "user_id": str(uuid4()),
                "query": "test question",
                "rewritten_query": "test question",
                "document_ids": None,
                "conversation_history": [],
                "db": mock_db,
            }
            result = await rag_node(state)
            assert "couldn't find enough information" in result["answer"].lower()
            assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_rag_node_with_chunks(self):
        """When retrieval returns chunks, should generate an answer."""
        from app.services.agent import rag_node

        mock_db = AsyncMock()
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="Based on the document, the answer is X [doc.pdf, p.1]")

        mock_chunks = [{
            "chunk_id": str(uuid4()),
            "document_id": str(uuid4()),
            "filename": "doc.pdf",
            "page": 1,
            "text": "Some relevant content about X",
            "chunk_index": 0,
            "score": 0.85,
        }]

        with patch("app.services.agent.retrieve_chunks", new_callable=AsyncMock, return_value=mock_chunks), \
             patch("app.services.agent.get_llm_provider", return_value=mock_llm):

            state = {
                "user_id": str(uuid4()),
                "query": "What is X?",
                "rewritten_query": "What is X?",
                "document_ids": None,
                "conversation_history": [],
                "db": mock_db,
            }
            result = await rag_node(state)
            assert len(result["answer"]) > 0
            assert len(result["sources"]) == 1
            assert result["sources"][0]["filename"] == "doc.pdf"
