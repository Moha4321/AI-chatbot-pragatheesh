
"""
test_embedding_engine.py
========================
Tests for the MPS-accelerated EmbeddingEngine.

Note: Running these tests will download the ~90MB all-MiniLM-L6-v2 model 
on the first run and cache it locally.
"""

import math
import numpy as np
import pytest

from embedding_engine import EmbeddingEngine

@pytest.fixture(scope="module")
def engine():
    """
    Load the engine once for the entire test module to save time.
    """
    return EmbeddingEngine()


class TestEmbeddingEngine:

    def test_initialization_shapes(self, engine):
        """Test that pre-computed matrices have the correct dimensions."""
        kb_embeddings, kb_texts = engine.get_kb_data()
        
        assert len(kb_texts) > 0, "KB texts should not be empty"
        assert kb_embeddings.shape[0] == len(kb_texts), "Row count must match number of facts"
        assert kb_embeddings.shape[1] == 384, "all-MiniLM-L6-v2 dimension must be 384"
        
        anchor = engine.get_junk_anchor()
        assert anchor.shape == (384,), "Anchor must be a 1D vector of length 384"

    def test_kb_embeddings_are_normalized(self, engine):
        """Test that all rows in the pre-computed KB matrix have an L2 norm of 1.0."""
        kb_embeddings, _ = engine.get_kb_data()
        norms = np.linalg.norm(kb_embeddings, axis=1)
        
        assert np.allclose(norms, 1.0, atol=1e-5), "Not all KB embeddings are unit-normalized"

    def test_anchor_is_normalized(self, engine):
        """Test that the junk food centroid is unit-normalized."""
        anchor = engine.get_junk_anchor()
        norm = float(np.linalg.norm(anchor))
        
        assert math.isclose(norm, 1.0, abs_tol=1e-5), f"Anchor norm is {norm}, expected 1.0"

    def test_get_embedding(self, engine):
        """Test embedding a new query string."""
        query = "I am craving some salty snacks"
        vector = engine.get_embedding(query)
        
        assert isinstance(vector, np.ndarray), "Should return a numpy array"
        assert vector.shape == (384,), "Should be a 1D array of shape 384"
        assert vector.dtype == np.float32, "Should be float32 for fast computation"
        
        # Must be normalized
        norm = float(np.linalg.norm(vector))
        assert math.isclose(norm, 1.0, abs_tol=1e-5), "Query vector must be unit-normalized"

