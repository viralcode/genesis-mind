"""
Genesis Mind — Response Decoder (Neural Voice)

The Personality GRU (Layer 3) produces a 64-dim response embedding
every time it processes an experience. This embedding is the network's
"own voice" — what it wants to say purely from its neural weights,
without any LLM.

But a 64-dim vector is not text. This decoder translates the neural
response back into human language by finding the closest known
concept(s) in semantic memory.

Method:
    1. Take the 64-dim response embedding from the GRU
    2. Compare it to all text_embeddings in SemanticMemory (cosine sim)
    3. Return the top-K closest concepts as the "neural voice"

This is NOT a language model. It is a lookup decoder that lets the
neural weights express themselves through the vocabulary Genesis has
learned. As Genesis learns more words, its neural voice becomes richer.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("genesis.neural.response_decoder")


class ResponseDecoder:
    """
    Decodes the GRU's response embedding into text by mapping
    it to the nearest known concepts in semantic memory.
    """

    def __init__(self, top_k: int = 3, min_similarity: float = 0.1):
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._decode_count = 0

        logger.info("Response decoder initialized (top_k=%d)", top_k)

    def decode(self, response_embedding: np.ndarray,
               semantic_memory) -> str:
        """
        Decode a response embedding into text.

        Finds the closest known concepts to the response vector
        and constructs a phrase from them.

        Args:
            response_embedding: 64-dim vector from the GRU output head
            semantic_memory: The SemanticMemory instance

        Returns:
            A string representing what the neural network "wants to say"
        """
        if response_embedding is None:
            return "(silence)"

        response = np.array(response_embedding, dtype=np.float32).flatten()

        # Get all concepts with text embeddings
        concepts = semantic_memory.get_all_concepts()
        if not concepts:
            return "(no words yet)"

        # Score each concept by cosine similarity to the response vector
        scored: List[Tuple[str, float]] = []
        for concept in concepts:
            if concept.text_embedding is not None:
                stored = np.array(concept.text_embedding, dtype=np.float32).flatten()

                # Project to same dim if needed (response is 64d, text_emb is 384d)
                # Use dot product of first min(len) dimensions as rough similarity
                min_dim = min(len(response), len(stored))
                r_proj = response[:min_dim]
                s_proj = stored[:min_dim]

                norm_r = np.linalg.norm(r_proj)
                norm_s = np.linalg.norm(s_proj)
                if norm_r > 0 and norm_s > 0:
                    sim = float(np.dot(r_proj, s_proj) / (norm_r * norm_s))
                    if sim >= self._min_similarity:
                        scored.append((concept.word, sim))

        if not scored:
            return "(searching for words...)"

        # Sort by similarity (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:self._top_k]

        self._decode_count += 1

        # Build the neural utterance
        words = [word for word, _ in top]
        confidences = [sim for _, sim in top]

        if len(words) == 1:
            return words[0]
        elif len(words) == 2:
            return f"{words[0]}... {words[1]}"
        else:
            return " — ".join(words)

    def decode_with_scores(self, response_embedding: np.ndarray,
                           semantic_memory) -> List[Tuple[str, float]]:
        """
        Like decode() but returns the raw scored concept list.

        Useful for introspection: see exactly what the neural
        network is "thinking about."
        """
        if response_embedding is None:
            return []

        response = np.array(response_embedding, dtype=np.float32).flatten()
        concepts = semantic_memory.get_all_concepts()
        if not concepts:
            return []

        scored: List[Tuple[str, float]] = []
        for concept in concepts:
            if concept.text_embedding is not None:
                stored = np.array(concept.text_embedding, dtype=np.float32).flatten()
                min_dim = min(len(response), len(stored))
                r_proj = response[:min_dim]
                s_proj = stored[:min_dim]
                norm_r = np.linalg.norm(r_proj)
                norm_s = np.linalg.norm(s_proj)
                if norm_r > 0 and norm_s > 0:
                    sim = float(np.dot(r_proj, s_proj) / (norm_r * norm_s))
                    scored.append((concept.word, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self._top_k * 2]

    def get_stats(self) -> dict:
        return {
            "decode_count": self._decode_count,
            "top_k": self._top_k,
        }

    def __repr__(self) -> str:
        return f"ResponseDecoder(decoded={self._decode_count}, top_k={self._top_k})"
