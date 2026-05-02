
"""
embedding_engine.py
===================
Semantic vector pipeline optimized for Apple Silicon (M4).

This module manages the all-MiniLM-L6-v2 sentence transformer. It is designed
as a singleton-like class to be instantiated once during server lifespan.

Hardware Co-Design (Thesis Note):
---------------------------------
By forcing device="mps", we route tensor operations through the M4's GPU 
via Metal Performance Shaders. Furthermore, we pre-compute and L2-normalize
the entire knowledge base and the junk-food centroid at boot time. This 
reduces all subsequent retrieval operations to pure O(1) matrix multiplications 
handled by `vector_math.py`.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Import our custom math primitives
from vector_math import l2_normalize, l2_normalize_matrix, compute_centroid

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent / "knowledge_base.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Canonical phrases used to define the "junk food / craving" semantic region
JUNK_FOOD_ANCHOR_PHRASES =[
    "I really want some chips",
    "I am craving chocolate right now",
    "Need a burger and fries",
    "I can't stop thinking about pizza",
    "I want to eat a whole tub of ice cream",
    "Craving sweet junk food",
    "I feel like bingeing on snacks"
]


class EmbeddingEngine:
    def __init__(self, model_name: str = MODEL_NAME, kb_path: Path = KB_PATH):
        """
        Initialise the embedding engine, load the model to MPS, and pre-compute
        all required vector matrices.
        """
        # 1. Hardware Selection
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Hardware acceleration enabled: Apple MPS (Metal Performance Shaders)")
        else:
            self.device = "cpu"
            logger.warning("MPS not found. Falling back to CPU.")

        # 2. Load Model
        logger.info(f"Loading SentenceTransformer model: {model_name} onto {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_embedding_dimension()

        # 3. Load Knowledge Base Texts
        self.kb_texts = self._load_kb_texts(kb_path)
        
        # 4. Pre-compute and Normalize KB Embeddings
        logger.info("Pre-computing knowledge base embeddings...")
        raw_kb_embeddings = self.model.encode(
            self.kb_texts, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        # We normalize here so cosine_similarity later is just a dot product
        self.kb_embeddings = l2_normalize_matrix(raw_kb_embeddings)
        logger.info(f"KB Matrix shape: {self.kb_embeddings.shape}")

        # 5. Pre-compute the Junk Food Centroid (v_junk)
        logger.info("Computing junk food anchor centroid...")
        raw_anchor_embeddings = self.model.encode(
            JUNK_FOOD_ANCHOR_PHRASES, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        # Average the canonical phrases, then L2-normalize the resulting vector
        centroid = compute_centroid(raw_anchor_embeddings)
        self.junk_anchor = l2_normalize(centroid)
        
        logger.info("EmbeddingEngine initialization complete.")

    def _load_kb_texts(self, path: Path) -> List[str]:
        """Read the local knowledge_base.json and extract the text strings."""
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found at {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        texts = [item["text"] for item in data if "text" in item]
        if not texts:
            raise ValueError("Knowledge base is empty or malformed.")
            
        return texts

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Embed a single user query and return its L2-normalized vector.
        
        Parameters
        ----------
        text : str
            The user's raw input message.
            
        Returns
        -------
        np.ndarray, shape (D,)
            The 384-D normalized semantic vector.
        """
        raw_embedding = self.model.encode(text, convert_to_numpy=True)
        return l2_normalize(raw_embedding)

    def get_kb_data(self) -> Tuple[np.ndarray, List[str]]:
        """Return the pre-computed KB embeddings matrix and parallel text list."""
        return self.kb_embeddings, self.kb_texts

    def get_junk_anchor(self) -> np.ndarray:
        """Return the pre-computed v_junk centroid vector."""
        return self.junk_anchor
