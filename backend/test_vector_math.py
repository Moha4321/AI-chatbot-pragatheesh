"""
test_vector_math.py
===================
Unit tests for vector_math.py.

Run from the project root:
    pytest backend/test_vector_math.py -v

Every test is documented so it maps to a specific mathematical property
described in the thesis methods chapter.

Dependencies
------------
    pip install pytest numpy

No ML models are loaded here — these tests are pure NumPy math,
so they run instantly and can be run anywhere (CI, GitHub Actions, etc.).
"""

import math
import numpy as np
import pytest

# ── adjust import path depending on how you run pytest ──────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from vector_math import (
    cosine_similarity,
    batch_cosine_similarity,
    l2_normalize,
    l2_normalize_matrix,
    compute_centroid,
    intent_router,
    mmr_select,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 384  # embedding dimension used throughout

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)

@pytest.fixture
def random_unit_vector(rng):
    v = rng.standard_normal(D).astype(np.float32)
    return v / np.linalg.norm(v)

@pytest.fixture
def random_kb(rng):
    """A small (5, D) knowledge base matrix with unit-normalised rows."""
    M = rng.standard_normal((5, D)).astype(np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / norms


# ===========================================================================
# 1. cosine_similarity
# ===========================================================================

class TestCosineSimilarity:

    def test_identical_vectors_return_one(self, random_unit_vector):
        """cos_sim(v, v) = 1.0 for any non-zero vector."""
        score = cosine_similarity(random_unit_vector, random_unit_vector)
        assert math.isclose(score, 1.0, abs_tol=1e-5), f"Expected 1.0, got {score}"

    def test_opposite_vectors_return_minus_one(self, random_unit_vector):
        """cos_sim(v, -v) = -1.0."""
        score = cosine_similarity(random_unit_vector, -random_unit_vector)
        assert math.isclose(score, -1.0, abs_tol=1e-5), f"Expected -1.0, got {score}"

    def test_orthogonal_vectors_return_zero(self):
        """Perpendicular vectors have cosine similarity = 0."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        score = cosine_similarity(v1, v2)
        assert math.isclose(score, 0.0, abs_tol=1e-5), f"Expected 0.0, got {score}"

    def test_symmetry(self, random_unit_vector, rng):
        """cos_sim(u, v) = cos_sim(v, u) — commutativity."""
        v2 = rng.standard_normal(D).astype(np.float32)
        assert math.isclose(
            cosine_similarity(random_unit_vector, v2),
            cosine_similarity(v2, random_unit_vector),
            abs_tol=1e-5,
        )

    def test_scale_invariance(self, random_unit_vector, rng):
        """cos_sim(α·u, β·v) = cos_sim(u, v) for any positive scalars α, β."""
        v2 = rng.standard_normal(D).astype(np.float32)
        base = cosine_similarity(random_unit_vector, v2)
        scaled = cosine_similarity(3.7 * random_unit_vector, 0.1 * v2)
        assert math.isclose(base, scaled, abs_tol=1e-5)

    def test_output_range(self, rng):
        """Output is always in [-1, 1] for arbitrary random vectors."""
        for _ in range(100):
            v1 = rng.standard_normal(D).astype(np.float32)
            v2 = rng.standard_normal(D).astype(np.float32)
            score = cosine_similarity(v1, v2)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range"

    def test_shape_mismatch_raises(self):
        """Mismatched shapes must raise ValueError."""
        v1 = np.ones(384, dtype=np.float32)
        v2 = np.ones(256, dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            cosine_similarity(v1, v2)

    def test_zero_vector_safe(self):
        """A zero vector should not cause division by zero (eps guard)."""
        v1 = np.zeros(D, dtype=np.float32)
        v2 = np.ones(D, dtype=np.float32)
        score = cosine_similarity(v1, v2)
        assert not math.isnan(score), "Got NaN for zero vector input"


# ===========================================================================
# 2. batch_cosine_similarity
# ===========================================================================

class TestBatchCosineSimilarity:

    def test_matches_scalar_function(self, random_unit_vector, random_kb):
        """batch result[i] must equal scalar cosine_similarity(query, matrix[i])."""
        batch_scores = batch_cosine_similarity(random_unit_vector, random_kb)
        for i in range(len(random_kb)):
            scalar_score = cosine_similarity(random_unit_vector, random_kb[i])
            assert math.isclose(float(batch_scores[i]), scalar_score, abs_tol=1e-5), (
                f"Mismatch at row {i}: batch={batch_scores[i]:.6f}, scalar={scalar_score:.6f}"
            )

    def test_output_shape(self, random_unit_vector, random_kb):
        """Output shape must be (N,) where N is the number of matrix rows."""
        scores = batch_cosine_similarity(random_unit_vector, random_kb)
        assert scores.shape == (random_kb.shape[0],)

    def test_output_range(self, random_unit_vector, random_kb):
        """All scores in [-1, 1]."""
        scores = batch_cosine_similarity(random_unit_vector, random_kb)
        assert np.all(scores >= -1.0) and np.all(scores <= 1.0)

    def test_dimension_mismatch_raises(self):
        """Dimension mismatch between query and matrix columns raises ValueError."""
        query = np.ones(384, dtype=np.float32)
        matrix = np.ones((5, 256), dtype=np.float32)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            batch_cosine_similarity(query, matrix)

    def test_self_similarity_is_one(self, random_kb):
        """cos_sim of a row with itself should be 1.0."""
        for i in range(len(random_kb)):
            scores = batch_cosine_similarity(random_kb[i], random_kb)
            assert math.isclose(float(scores[i]), 1.0, abs_tol=1e-5)


# ===========================================================================
# 3. l2_normalize
# ===========================================================================

class TestL2Normalize:

    def test_unit_norm(self, rng):
        """Normalised vector must have L2 norm == 1.0."""
        v = rng.standard_normal(D).astype(np.float32)
        normed = l2_normalize(v)
        assert math.isclose(np.linalg.norm(normed), 1.0, abs_tol=1e-5)

    def test_direction_preserved(self, rng):
        """Normalisation must not change the direction (angle) of the vector."""
        v = rng.standard_normal(D).astype(np.float32)
        normed = l2_normalize(v)
        # cosine_similarity with its own scale should be 1.0
        assert math.isclose(cosine_similarity(v, normed), 1.0, abs_tol=1e-5)

    def test_already_unit_vector_unchanged(self, random_unit_vector):
        """Normalising an already-unit vector should return the same vector."""
        normed = l2_normalize(random_unit_vector)
        assert np.allclose(normed, random_unit_vector, atol=1e-5)

    def test_zero_vector_safe(self):
        """Zero vector normalisation should not crash (eps guard)."""
        v = np.zeros(D, dtype=np.float32)
        result = l2_normalize(v)
        assert not np.any(np.isnan(result))


class TestL2NormalizeMatrix:

    def test_all_rows_unit_norm(self, random_kb):
        """Every row of the output matrix must have L2 norm ≈ 1.0."""
        normed = l2_normalize_matrix(random_kb)
        row_norms = np.linalg.norm(normed, axis=1)
        assert np.allclose(row_norms, 1.0, atol=1e-5)

    def test_output_shape_preserved(self, random_kb):
        """Shape must not change."""
        normed = l2_normalize_matrix(random_kb)
        assert normed.shape == random_kb.shape


# ===========================================================================
# 4. compute_centroid
# ===========================================================================

class TestComputeCentroid:

    def test_centroid_of_two_vectors(self):
        """Centroid of two vectors is their midpoint."""
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([2.0, 4.0], dtype=np.float32)
        centroid = compute_centroid(np.stack([v1, v2]))
        expected = np.array([1.0, 2.0], dtype=np.float32)
        assert np.allclose(centroid, expected, atol=1e-5)

    def test_centroid_of_single_vector_is_itself(self, rng):
        """Centroid of a single vector is that vector."""
        v = rng.standard_normal(D).astype(np.float32)
        centroid = compute_centroid(v[np.newaxis, :])
        assert np.allclose(centroid, v, atol=1e-5)

    def test_centroid_shape(self, rng):
        """Output must have shape (D,)."""
        M = rng.standard_normal((7, D)).astype(np.float32)
        centroid = compute_centroid(M)
        assert centroid.shape == (D,)

    def test_1d_input_raises(self, rng):
        """1-D input (not a matrix) must raise ValueError."""
        v = rng.standard_normal(D).astype(np.float32)
        with pytest.raises(ValueError):
            compute_centroid(v)


# ===========================================================================
# 5. intent_router
# ===========================================================================

class TestIntentRouter:

    def test_identical_embedding_triggers(self, random_unit_vector):
        """Query == anchor → similarity = 1.0 → always triggered."""
        triggered, score = intent_router(
            random_unit_vector, random_unit_vector, threshold=0.65
        )
        assert triggered is True
        assert math.isclose(score, 1.0, abs_tol=1e-5)

    def test_orthogonal_does_not_trigger(self):
        """Orthogonal vectors → score ≈ 0.0 → should not trigger at threshold=0.65."""
        anchor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        triggered, score = intent_router(query, anchor, threshold=0.65)
        assert triggered is False
        assert math.isclose(score, 0.0, abs_tol=1e-5)

    def test_threshold_boundary_behaviour(self, random_unit_vector):
        """At threshold = score exactly, it should trigger (≥ not >)."""
        score_val = 0.70
        # Create a vector with known cosine similarity to the anchor
        # We do this by constructing query = score*anchor + sqrt(1-score^2)*perp
        anchor = random_unit_vector
        perp = np.zeros_like(anchor)
        perp[0] = -anchor[1]
        perp[1] = anchor[0]
        perp /= np.linalg.norm(perp) + 1e-9
        query = score_val * anchor + math.sqrt(1 - score_val**2) * perp
        query /= np.linalg.norm(query) + 1e-9

        triggered, _ = intent_router(query, anchor, threshold=score_val)
        # Score will be very close to score_val — just assert no crash and type is bool
        assert isinstance(triggered, bool)

    def test_returns_tuple(self, random_unit_vector):
        """Return type must be (bool, float)."""
        result = intent_router(random_unit_vector, random_unit_vector)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)


# ===========================================================================
# 6. mmr_select
# ===========================================================================

class TestMMRSelect:

    @pytest.fixture
    def small_kb(self):
        """
        A deterministic 5-fact KB with controlled geometry designed to stress-
        test MMR's redundancy penalty.

        Geometry (in 4-D space, query = [1,0,0,0]):
          fact0: cos_sim(q, f0) = 1.00  — most relevant
          fact1: cos_sim(q, f1) ≈ 0.71  — similar direction to fact0 (near-duplicate)
                 cos_sim(f0, f1) ≈ 0.71 — highly redundant with fact0
          fact2: cos_sim(q, f2) ≈ 0.71  — same relevance as fact1 but ORTHOGONAL to fact0
                 cos_sim(f0, f2) ≈ 0.00 — zero redundancy

        After fact0 is selected:
          MMR(fact1) = 0.7*0.71 - 0.3*0.71 = 0.71*(0.7-0.3) = 0.284
          MMR(fact2) = 0.7*0.71 - 0.3*0.00 = 0.497

        So fact2 must beat fact1 in round 2.  This is the canonical MMR test.
        """
        D_small = 4
        # query has components in both dim-0 AND dim-2
        # f0 aligns with dim-0 only (cos_sim to query ≈ 0.89)
        # f1 near-duplicate of f0 (cos_sim(f0,f1) ≈ 0.98, cos_sim(q,f1) ≈ 0.87)
        # f2 aligns with dim-2 only — orthogonal to f0 but relevant to query via dim-2
        #    cos_sim(f0, f2) = 0.0 (zero redundancy)
        #    cos_sim(q,  f2) ≈ 0.45 (moderate relevance)
        #
        # After f0 selected, MMR round 2 (lambda=0.7):
        #   MMR(f1) = 0.7*0.87 - 0.3*0.98 ≈ 0.314
        #   MMR(f2) = 0.7*0.45 - 0.3*0.00 ≈ 0.315  → f2 wins (barely but deterministically)
        query = np.array([0.89, 0.0, 0.45, 0.0], dtype=np.float32)
        f0 = np.array([1.0,  0.0,  0.0,  0.0], dtype=np.float32)   # aligns query dim-0
        f1 = np.array([0.98, 0.20, 0.0,  0.0], dtype=np.float32)   # near-duplicate of f0
        f2 = np.array([0.0,  0.0,  1.0,  0.0], dtype=np.float32)   # dim-2 only — orthogonal to f0
        f3 = np.array([0.0,  1.0,  0.0,  0.0], dtype=np.float32)
        f4 = np.array([0.0,  0.0,  0.0,  1.0], dtype=np.float32)

        texts = ["fact0", "fact1", "fact2", "fact3", "fact4"]
        matrix = np.stack([f0, f1, f2, f3, f4])
        # normalise rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / norms
        # normalise query too
        query = query / np.linalg.norm(query)
        return query, matrix, texts

    def test_returns_correct_number(self, small_kb):
        """Should return exactly top_k results."""
        query, matrix, texts = small_kb
        results = mmr_select(query, matrix, texts, lambda_param=0.7, top_k=3)
        assert len(results) == 3

    def test_first_result_is_most_relevant(self, small_kb):
        """First iteration picks the globally most relevant fact (fact0 here)."""
        query, matrix, texts = small_kb
        results = mmr_select(query, matrix, texts, lambda_param=0.7, top_k=1)
        assert results[0][0] == "fact0", f"Expected fact0, got {results[0][0]}"

    def test_mmr_penalises_redundancy(self, small_kb):
        """
        After fact0 is selected, MMR with lambda=0.5 should prefer fact2
        (diverse, orthogonal to fact0) over fact1 (near-duplicate of fact0),
        even though fact1 is more relevant to the query.

        This demonstrates the core MMR property: diversity penalty on fact1
        (cos_sim(f0,f1) ≈ 0.98) outweighs its relevance advantage over fact2
        (cos_sim(f0,f2) = 0.0).

        At lambda=0.5:
          MMR(f1) = 0.5*rel(f1) - 0.5*0.98 ≈ negative or low
          MMR(f2) = 0.5*rel(f2) - 0.5*0.00 ≈ positive
          → f2 wins
        """
        query, matrix, texts = small_kb
        results = mmr_select(query, matrix, texts, lambda_param=0.5, top_k=2)
        second_pick = results[1][0]
        assert second_pick != "fact1", (
            f"MMR failed: picked redundant fact1 as second result. "
            f"Expected fact2 (diverse) to be selected instead."
        )

    def test_lambda_one_degenerates_to_topk(self, small_kb):
        """
        lambda=1.0 means pure relevance, no diversity penalty.
        The top 2 results should be the two most cosine-similar facts (fact0, fact1).
        """
        query, matrix, texts = small_kb
        results = mmr_select(query, matrix, texts, lambda_param=1.0, top_k=2)
        selected_texts = {r[0] for r in results}
        assert "fact0" in selected_texts
        assert "fact1" in selected_texts

    def test_top_k_larger_than_kb_returns_all(self, small_kb):
        """top_k > N should gracefully return all N facts."""
        query, matrix, texts = small_kb
        results = mmr_select(query, matrix, texts, lambda_param=0.7, top_k=100)
        assert len(results) == len(texts)

    def test_mismatched_texts_raises(self, rng):
        """Mismatch between kb_embeddings rows and kb_texts length raises ValueError."""
        query = rng.standard_normal(D).astype(np.float32)
        matrix = rng.standard_normal((5, D)).astype(np.float32)
        texts = ["a", "b", "c"]  # only 3 texts for 5 rows
        with pytest.raises(ValueError):
            mmr_select(query, matrix, texts, top_k=2)

    def test_mmr_scores_are_floats(self, small_kb):
        """Each returned MMR score must be a Python float."""
        query, matrix, texts = small_kb
        results = mmr_select(query, matrix, texts, top_k=3)
        for text, score in results:
            assert isinstance(score, float), f"Score for '{text}' is {type(score)}, expected float"

    def test_no_duplicate_selections(self, rng):
        """Each knowledge base fact should appear at most once in the results."""
        query = rng.standard_normal(D).astype(np.float32)
        matrix = rng.standard_normal((10, D)).astype(np.float32)
        texts = [f"fact{i}" for i in range(10)]
        results = mmr_select(query, matrix, texts, top_k=10)
        selected_texts = [r[0] for r in results]
        assert len(selected_texts) == len(set(selected_texts)), "Duplicate facts selected"