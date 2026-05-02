"""
vector_math.py
==============
Core mathematical primitives for the RAG pipeline.

All functions are implemented from first principles using NumPy — deliberately
avoiding sklearn or scipy wrappers — so that every operation maps directly to
a derivable formula in the thesis methods chapter.

Mathematical background
-----------------------
Embeddings produced by all-MiniLM-L6-v2 live in R^384.  We need three
operations:

  1. Cosine similarity   — measures semantic angle between two vectors.
  2. Batch cosine sim    — vectorised form over a (N, 384) matrix.
  3. MMR selection       — greedily selects a diverse-yet-relevant subset.
  4. Intent routing      — thresholds similarity against a junk-food centroid.

"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Vector = np.ndarray   # shape (D,)  — a single embedding
Matrix = np.ndarray   # shape (N, D) — N embeddings stacked as rows

# ---------------------------------------------------------------------------
# 1.  Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(v1: Vector, v2: Vector, eps: float = 1e-9) -> float:
    """
    Compute the cosine similarity between two dense vectors.

    Definition
    ----------
    Given vectors u, v ∈ R^D:

        cos_sim(u, v) = (u · v) / (‖u‖₂ · ‖v‖₂)

    The result lies in [-1, 1].  For unit-normalised embedding models
    (which all-MiniLM-L6-v2 produces after mean pooling + L2 norm), the
    denominator is always 1, so this reduces to a dot product — but we
    keep the full formula for correctness when inputs are not pre-normalised.

    Parameters
    ----------
    v1, v2 : np.ndarray, shape (D,)
        Dense embedding vectors.
    eps : float
        Small constant to guard against division by zero for zero vectors.

    Returns
    -------
    float
        Cosine similarity score in [-1, 1].

    Raises
    ------
    ValueError
        If v1 and v2 have different shapes.
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    if v1.shape != v2.shape:
        raise ValueError(
            f"Shape mismatch: v1 is {v1.shape}, v2 is {v2.shape}. "
            "Both vectors must have the same dimensionality."
        )

    dot_product: float = float(np.dot(v1, v2))
    norm_v1: float = float(np.linalg.norm(v1))
    norm_v2: float = float(np.linalg.norm(v2))
    denominator: float = norm_v1 * norm_v2 + eps

    similarity: float = dot_product / denominator
    # Clamp to [-1, 1] to handle any floating-point drift
    return float(np.clip(similarity, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 2.  Batch Cosine Similarity
# ---------------------------------------------------------------------------

def batch_cosine_similarity(query: Vector, matrix: Matrix, eps: float = 1e-9) -> Vector:
    """
    Compute cosine similarity between one query vector and every row of a matrix.

    This is the vectorised form of cosine_similarity, operating on an entire
    knowledge-base matrix in a single BLAS call instead of N sequential calls.

    Mathematical form
    -----------------
    Let q ∈ R^D be the query, M ∈ R^{N×D} be the matrix of N knowledge-base
    embeddings.

    Step 1 — Compute all dot products simultaneously:
        scores = M @ q           ∈ R^N

    Step 2 — Normalise:
        scores = scores / (‖M‖_row · ‖q‖₂)

    where ‖M‖_row ∈ R^N is the L2 norm of each row of M.

    Result: scores[i] = cos_sim(q, M[i])

    Complexity
    ----------
    O(N·D) — a single matrix-vector multiply.  For N=20, D=384 this is
    negligible (<1 ms on MPS).  The function is written to scale to N=10^6
    if the knowledge base grows.

    Parameters
    ----------
    query : np.ndarray, shape (D,)
    matrix : np.ndarray, shape (N, D)
    eps : float

    Returns
    -------
    np.ndarray, shape (N,)
        Cosine similarity of query against each row.
    """
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError(f"query must be 1-D, got shape {query.shape}")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query is {query.shape[0]}-D, "
            f"matrix columns are {matrix.shape[1]}-D."
        )

    # Row-wise L2 norms of the matrix  →  shape (N,)
    row_norms: Vector = np.linalg.norm(matrix, axis=1)          # shape (N,)
    query_norm: float = float(np.linalg.norm(query))

    # All dot products in one call  →  shape (N,)
    dot_products: Vector = matrix @ query                         # shape (N,)

    denominators: Vector = row_norms * query_norm + eps           # shape (N,)
    similarities: Vector = dot_products / denominators            # shape (N,)

    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3.  L2 Normalisation (utility)
# ---------------------------------------------------------------------------

def l2_normalize(v: Vector, eps: float = 1e-9) -> Vector:
    """
    Return the L2-normalised form of a vector: v̂ = v / ‖v‖₂

    After normalisation, cosine_similarity(v̂1, v̂2) = v̂1 · v̂2 (pure dot
    product).  This is useful when you want to pre-normalise all KB embeddings
    once to make subsequent similarity lookups even cheaper.

    Parameters
    ----------
    v : np.ndarray, shape (D,)
    eps : float

    Returns
    -------
    np.ndarray, shape (D,)
    """
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v) + eps
    return (v / norm).astype(np.float32)


def l2_normalize_matrix(matrix: Matrix, eps: float = 1e-9) -> Matrix:
    """
    Row-wise L2 normalisation of a (N, D) matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, D)
    eps : float

    Returns
    -------
    np.ndarray, shape (N, D)
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + eps  # (N, 1)
    return (matrix / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# 4.  Centroid Computation
# ---------------------------------------------------------------------------

def compute_centroid(vectors: Matrix) -> Vector:
    """
    Compute the arithmetic mean (centroid) of a set of vectors.

    Used to build the v_junk anchor: given K canonical junk-food query
    embeddings, their centroid is a single representative point in embedding
    space that captures the "junk food intent" region.

    Parameters
    ----------
    vectors : np.ndarray, shape (K, D)
        K example embeddings to average.

    Returns
    -------
    np.ndarray, shape (D,)
        The mean vector (not normalised — caller may normalise if desired).
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {vectors.shape}")
    return np.mean(vectors, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 5.  Intent Router
# ---------------------------------------------------------------------------

def intent_router(
    query_embedding: Vector,
    anchor_embedding: Vector,
    threshold: float = 0.65,
) -> Tuple[bool, float]:
    """
    Threshold-based intent classifier using cosine similarity.

    Determines whether a user query is semantically close enough to the
    junk-food anchor centroid (v_junk) to warrant RAG retrieval.

    Decision rule
    -------------
        triggered = cos_sim(query, v_junk) ≥ threshold

    Threshold selection
    -------------------
    The default value of 0.65 is a starting heuristic.  In the thesis, this
    should be calibrated empirically:

      1. Manually annotate 50–100 test queries as {junk_intent, other}.
      2. Compute cos_sim(query, v_junk) for each.
      3. Sweep threshold ∈ [0.40, 0.90] in steps of 0.05.
      4. Select the threshold that maximises F1 on the annotated set.
      5. Report the precision-recall curve as a thesis figure.

    Parameters
    ----------
    query_embedding : np.ndarray, shape (D,)
    anchor_embedding : np.ndarray, shape (D,)
        The pre-computed v_junk centroid.
    threshold : float
        Decision boundary.  Queries with similarity ≥ threshold trigger RAG.

    Returns
    -------
    (triggered: bool, score: float)
        triggered — whether RAG should be activated.
        score     — the raw similarity score (logged to telemetry CSV).
    """
    score: float = cosine_similarity(query_embedding, anchor_embedding)
    triggered: bool = score >= threshold

    logger.debug(
        "intent_router | score=%.4f | threshold=%.2f | triggered=%s",
        score, threshold, triggered,
    )
    return triggered, score


# ---------------------------------------------------------------------------
# 6.  Maximal Marginal Relevance (MMR)
# ---------------------------------------------------------------------------

def mmr_select(
    query_embedding: Vector,
    kb_embeddings: Matrix,
    kb_texts: List[str],
    lambda_param: float = 0.7,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Maximal Marginal Relevance selection over a knowledge base.

    MMR (Carbonell & Goldstein, 1998) greedily selects documents that are
    simultaneously relevant to the query and diverse relative to already-
    selected documents.

    Algorithm
    ---------
    At each iteration i, select the candidate c* that maximises:

        MMR(c) = λ · Rel(c, query)  −  (1−λ) · Redundancy(c, Selected)

    where:
        Rel(c, query)           = cos_sim(embedding(c), query_embedding)
        Redundancy(c, Selected) = max_{s ∈ Selected} cos_sim(embedding(c), embedding(s))

    Repeat for top_k iterations.

    Parameter interpretation
    ------------------------
    λ = 1.0  →  pure relevance ranking (degenerate to top-k cosine similarity)
    λ = 0.0  →  pure diversity (ignores query relevance entirely)
    λ = 0.7  →  recommended default: relevance-weighted with diversity penalty

    Thesis ablation study
    ---------------------
    Compare MMR (λ=0.7) against top-k (λ=1.0).  Metrics:
      - Intra-list similarity (ILS): lower is more diverse.
      - User-rated helpfulness of chatbot response (Likert 1-5).
      - Proportion of retrieved facts that are semantically redundant.

    Complexity
    ----------
    O(top_k · N · D) — for N=20, top_k=3, D=384 this is ~23,040 flops,
    completing in <0.1 ms on any modern CPU.

    Parameters
    ----------
    query_embedding : np.ndarray, shape (D,)
    kb_embeddings : np.ndarray, shape (N, D)
        Pre-computed embeddings for all N knowledge-base facts.
    kb_texts : List[str]
        The raw text of each knowledge-base fact (parallel to kb_embeddings).
    lambda_param : float
        Trade-off parameter λ ∈ [0, 1].
    top_k : int
        Number of facts to retrieve.

    Returns
    -------
    List of (fact_text, mmr_score) tuples, length min(top_k, N),
    ordered by selection sequence (most relevant-and-diverse first).

    Raises
    ------
    ValueError
        If kb_embeddings and kb_texts have different lengths.
    """
    kb_embeddings = np.asarray(kb_embeddings, dtype=np.float32)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    n_candidates = kb_embeddings.shape[0]

    if len(kb_texts) != n_candidates:
        raise ValueError(
            f"kb_texts has {len(kb_texts)} entries but kb_embeddings has "
            f"{n_candidates} rows. They must be the same length."
        )

    effective_k = min(top_k, n_candidates)

    # Pre-compute relevance scores for all candidates: shape (N,)
    relevance_scores: Vector = batch_cosine_similarity(query_embedding, kb_embeddings)

    selected_indices: List[int] = []
    selected_embeddings: List[Vector] = []
    results: List[Tuple[str, float]] = []

    remaining_indices = list(range(n_candidates))

    for iteration in range(effective_k):
        best_idx: int = -1
        best_mmr_score: float = -np.inf

        for idx in remaining_indices:
            relevance: float = float(relevance_scores[idx])

            # Redundancy term: max cosine sim to any already-selected embedding
            if len(selected_embeddings) == 0:
                # First iteration — no selected set yet, redundancy = 0
                redundancy: float = 0.0
            else:
                selected_matrix = np.stack(selected_embeddings, axis=0)  # (S, D)
                sim_to_selected: Vector = batch_cosine_similarity(
                    kb_embeddings[idx], selected_matrix
                )
                redundancy = float(np.max(sim_to_selected))

            mmr_score: float = (
                lambda_param * relevance
                - (1.0 - lambda_param) * redundancy
            )

            logger.debug(
                "  MMR iter=%d | idx=%d | rel=%.4f | red=%.4f | mmr=%.4f",
                iteration, idx, relevance, redundancy, mmr_score,
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        # Commit the best candidate
        selected_indices.append(best_idx)
        selected_embeddings.append(kb_embeddings[best_idx])
        results.append((kb_texts[best_idx], best_mmr_score))
        remaining_indices.remove(best_idx)

        logger.info(
            "MMR iter=%d | selected idx=%d | score=%.4f | text=%.60s...",
            iteration, best_idx, best_mmr_score, kb_texts[best_idx],
        )

    return results