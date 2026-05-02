"""
rag_engine.py
=============
Retrieval-Augmented Generation Orchestrator.

This module acts as the bridge between the semantic embedding space and the 
downstream LLM. It executes the core behavioural intervention logic:
  1. Is the user experiencing a craving? (Intent Routing)
  2. If so, what are the most diverse & relevant facts to help? (MMR Selection)
  3. Format these facts for prompt injection.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

from embedding_engine import EmbeddingEngine
from vector_math import intent_router, mmr_select

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    """Structured container for telemetry and LLM injection."""
    triggered: bool
    intent_score: float
    context_string: str
    selected_facts: List[Tuple[str, float]]  # List of (fact_text, mmr_score)


class RAGEngine:
    def __init__(
        self, 
        intent_threshold: float = 0.55, 
        mmr_lambda: float = 0.7, 
        top_k: int = 3
    ):
        """
        Initialize the RAG pipeline with thesis-defined hyperparameters.
        
        Parameters
        ----------
        intent_threshold : float
            Cosine similarity boundary. Scores >= this value trigger RAG.
        mmr_lambda : float
            Diversity tradeoff (1.0 = pure relevance, 0.0 = pure diversity).
        top_k : int
            Number of knowledge base facts to retrieve.
        """
        self.intent_threshold = intent_threshold
        self.mmr_lambda = mmr_lambda
        self.top_k = top_k
        logger.info(
            f"RAGEngine initialized | threshold={self.intent_threshold} | "
            f"lambda={self.mmr_lambda} | top_k={self.top_k}"
        )

    def process_query(self, query_text: str, embedder: EmbeddingEngine) -> RAGResult:
        """
        Process a raw text query through the semantic pipeline.
        
        Returns
        -------
        RAGResult
            Contains the boolean trigger, similarity score, and formatted context.
        """
        # 1. Embed the user's raw text
        query_vector = embedder.get_embedding(query_text)
        
        # 2. Check semantic proximity to the junk-food anchor
        junk_anchor = embedder.get_junk_anchor()
        triggered, intent_score = intent_router(
            query_vector, 
            junk_anchor, 
            threshold=self.intent_threshold
        )
        
        # 3. If not a craving, return an empty context safely
        if not triggered:
            return RAGResult(
                triggered=False,
                intent_score=intent_score,
                context_string="",
                selected_facts=[]
            )
            
        # 4. If it IS a craving, retrieve facts using Maximal Marginal Relevance
        kb_matrix, kb_texts = embedder.get_kb_data()
        selected_facts = mmr_select(
            query_embedding=query_vector,
            kb_embeddings=kb_matrix,
            kb_texts=kb_texts,
            lambda_param=self.mmr_lambda,
            top_k=self.top_k
        )
        
        # 5. Format the retrieved facts into a numbered list for the LLM
        context_string = self._format_context(selected_facts)
        
        return RAGResult(
            triggered=True,
            intent_score=intent_score,
            context_string=context_string,
            selected_facts=selected_facts
        )

    def _format_context(self, facts: List[Tuple[str, float]]) -> str:
        """Format the retrieved facts into a strict LLM-readable block."""
        if not facts:
            return ""
            
        formatted = "RELEVANT HEALTH FACTS FOR THIS USER:\n"
        for i, (text, score) in enumerate(facts, 1):
            formatted += f"{i}. {text}\n"
            
        return formatted.strip()
