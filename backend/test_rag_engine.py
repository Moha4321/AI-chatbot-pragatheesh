"""
test_rag_engine.py
==================
Unit tests verifying the behavioural logic gates of the RAG pipeline.
"""

import pytest
from embedding_engine import EmbeddingEngine
from rag_engine import RAGEngine

@pytest.fixture(scope="module")
def embedder():
    """Load the MPS embedder once for all tests."""
    return EmbeddingEngine()

@pytest.fixture
def rag():
    """Instantiate a RAG engine with default thesis parameters."""
    return RAGEngine(intent_threshold=0.55, mmr_lambda=0.7, top_k=3)


class TestRAGEngine:

    def test_non_craving_query_bypasses_rag(self, rag, embedder):
        """
        A casual conversation should NOT trigger the intervention logic.
        """
        query = "Hello, how are you today?"
        result = rag.process_query(query, embedder)
        
        assert result.triggered is False, "A greeting should not trigger RAG"
        assert result.intent_score < 0.60, "Intent score should be low"
        assert result.context_string == "", "Context string must be empty"
        assert len(result.selected_facts) == 0, "No facts should be selected"

    def test_craving_query_triggers_rag(self, rag, embedder):
        """
        A direct expression of a craving MUST trigger the intervention logic
        and retrieve exactly `top_k` facts.
        """
        query = "I am so stressed and I just want to eat a whole pizza"
        result = rag.process_query(query, embedder)
        
        assert result.triggered is True, "A direct craving should trigger RAG"
        assert result.intent_score >= 0.55, "Intent score should clear the threshold"
        assert len(result.selected_facts) == rag.top_k, f"Must retrieve {rag.top_k} facts"
        
        # Verify the context string is formatted correctly
        assert "RELEVANT HEALTH FACTS" in result.context_string
        assert "1." in result.context_string
        assert "2." in result.context_string

    def test_context_formatting(self, rag):
        """
        Ensure the string formatting method correctly numbers the inputs.
        """
        mock_facts =[
            ("Drinking water helps.", 0.85),
            ("Walking reduces cravings.", 0.65)
        ]
        context = rag._format_context(mock_facts)
        
        assert context == (
            "RELEVANT HEALTH FACTS FOR THIS USER:\n"
            "1. Drinking water helps.\n"
            "2. Walking reduces cravings."
        )
