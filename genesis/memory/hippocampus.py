"""
Genesis Mind — The Hippocampus

The core memory system. This is where all of Genesis's experiences
are stored as mathematical vectors in a local database.

In the human brain, the hippocampus converts short-term experiences
into long-term memories. In Genesis, ChromaDB serves this role:
every image embedding, every transcribed word, every learned concept
is stored here with rich metadata (timestamps, emotional valence,
source modality, developmental phase).

The hippocampus supports:
- STORAGE: Save a new experience as a vector with metadata
- RECALL:  Find similar memories given a query vector
- FORGET:  Prune weak memories during sleep consolidation
- COUNT:   Track how many memories exist (developmental metric)

No external server needed — ChromaDB runs entirely in-process.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
from collections import deque
import random

logger = logging.getLogger("genesis.memory.hippocampus")


class Hippocampus:
    """
    The vector memory database of Genesis.

    Stores all multimodal embeddings (visual, audio, text) in a
    local persistent ChromaDB instance. Supports similarity search,
    metadata filtering, and memory management.
    """

    def __init__(self, persist_dir: str, embedding_dim: int = 384):
        self.persist_dir = persist_dir
        self.embedding_dim = embedding_dim
        self._client = None
        self._collections: Dict[str, Any] = {}
        
        # Short-term replay buffer for neural weight consolidation (sleep/offline learning)
        self.replay_buffer = deque(maxlen=10000)

        self._initialize()
        logger.info("Hippocampus initialized at %s", persist_dir)

    def _initialize(self):
        """Initialize the ChromaDB client and core collections."""
        import chromadb
        from chromadb.config import Settings

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Core memory collections — each stores a different kind of memory
        collection_names = [
            "visual",       # Image embeddings from the eyes
            "auditory",     # Audio/text embeddings from the ears
            "concepts",     # Bound multimodal concept embeddings
            "episodes",     # Timestamped experience records
            "phonetics",    # Letter↔Sound binding records
        ]

        for name in collection_names:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )

        logger.info(
            "Memory collections ready: %s",
            ", ".join(collection_names),
        )

    def store(
        self,
        collection: str,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        document: str = "",
    ):
        """
        Store a single memory.

        Args:
            collection: Which collection to store in (visual, auditory, concepts, etc.)
            id: Unique identifier for this memory
            embedding: The vector representation (list of floats)
            metadata: Rich metadata (timestamp, source, emotional_valence, etc.)
            document: Optional text document associated with the memory
        """
        if collection not in self._collections:
            raise ValueError(f"Unknown collection: {collection}")

        # Ensure metadata values are simple types (ChromaDB requirement)
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)

        # Add timestamp if not present
        if "timestamp" not in clean_metadata:
            clean_metadata["timestamp"] = datetime.now().isoformat()

        self._collections[collection].upsert(
            ids=[id],
            embeddings=[embedding],
            metadatas=[clean_metadata],
            documents=[document] if document else None,
        )

        logger.debug("Stored memory '%s' in collection '%s'", id, collection)

    def recall(
        self,
        collection: str,
        query_embedding: List[float],
        n: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recall memories similar to the query.

        Returns the N most similar memories, ordered by similarity.
        Each result includes: id, distance, metadata, and document.
        """
        if collection not in self._collections:
            raise ValueError(f"Unknown collection: {collection}")

        coll = self._collections[collection]

        if coll.count() == 0:
            return []

        n = min(n, coll.count())
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = coll.query(**kwargs)

        # Flatten results into a list of dicts
        memories = []
        for i in range(len(results["ids"][0])):
            memories.append({
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1.0 - results["distances"][0][i],  # Cosine: lower distance = more similar
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "document": results["documents"][0][i] if results["documents"] else "",
            })

        return memories

    def recall_by_text(
        self,
        collection: str,
        query_text: str,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recall memories using text search on stored documents.
        """
        if collection not in self._collections:
            raise ValueError(f"Unknown collection: {collection}")

        coll = self._collections[collection]
        if coll.count() == 0:
            return []

        n = min(n, coll.count())
        results = coll.query(
            query_texts=[query_text],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for i in range(len(results["ids"][0])):
            memories.append({
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1.0 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "document": results["documents"][0][i] if results["documents"] else "",
            })

        return memories

    def forget(self, collection: str, ids: List[str]):
        """
        Forget specific memories (remove them permanently).

        Used during sleep consolidation to prune weak memories.
        """
        if collection not in self._collections:
            return
        try:
            self._collections[collection].delete(ids=ids)
            logger.info("Forgot %d memories from '%s'", len(ids), collection)
        except Exception as e:
            logger.error("Failed to forget memories: %s", e)

    def count(self, collection: str = None) -> int:
        """
        Count total memories. If collection is None, count all.
        """
        if collection:
            return self._collections.get(collection, None).count() if collection in self._collections else 0
        return sum(c.count() for c in self._collections.values())

    def get_all_ids(self, collection: str) -> List[str]:
        """Get all memory IDs in a collection."""
        if collection not in self._collections:
            return []
        result = self._collections[collection].get(include=[])
        return result["ids"]

    def get_memory(self, collection: str, id: str) -> Optional[Dict]:
        """Retrieve a specific memory by ID."""
        if collection not in self._collections:
            return None
        result = self._collections[collection].get(
            ids=[id],
            include=["documents", "metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "metadata": result["metadatas"][0] if result["metadatas"] else {},
            "document": result["documents"][0] if result["documents"] else "",
            "embedding": result["embeddings"][0] if result["embeddings"] else None,
        }

    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics across all collections."""
        return {name: coll.count() for name, coll in self._collections.items()}

    def add_to_replay(self, visual_latent: np.ndarray, auditory_latent: np.ndarray, limbic_state: Dict[str, float], concept_embedding: np.ndarray):
        """Add raw neural states to the short-term replay buffer."""
        self.replay_buffer.append({
            "visual": visual_latent,
            "auditory": auditory_latent,
            "limbic": limbic_state,
            "concept": concept_embedding
        })

    def sample_replay_batch(self, batch_size: int = 32) -> List[Dict]:
        """Sample a diverse batch of past experiences for neural consolidation."""
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
        return random.sample(self.replay_buffer, batch_size)

    def __repr__(self) -> str:
        total = self.count()
        return f"Hippocampus(total_memories={total}, path='{self.persist_dir}')"


# =============================================================================
# Standalone test — run with: python -m genesis.memory.hippocampus
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Hippocampus Test")
    print("Testing vector memory storage and recall...")
    print("=" * 60)

    import shutil
    test_dir = "/tmp/genesis_hippocampus_test"
    shutil.rmtree(test_dir, ignore_errors=True)

    hippo = Hippocampus(persist_dir=test_dir)

    # Store some concept embeddings
    dim = 384
    apple_emb = np.random.randn(dim).tolist()
    banana_emb = np.random.randn(dim).tolist()

    hippo.store("concepts", "concept_apple", apple_emb, {
        "word": "apple",
        "source": "teaching",
        "emotional_valence": "positive",
    }, document="A red fruit that grows on trees")

    hippo.store("concepts", "concept_banana", banana_emb, {
        "word": "banana",
        "source": "teaching",
        "emotional_valence": "positive",
    }, document="A yellow curved fruit")

    # Recall by similarity
    results = hippo.recall("concepts", apple_emb, n=2)
    print(f"\nRecall results for 'apple' query:")
    for r in results:
        print(f"  {r['id']}: similarity={r['similarity']:.4f}, doc='{r['document']}'")

    # Stats
    print(f"\nMemory stats: {hippo.get_stats()}")
    print(f"Total memories: {hippo.count()}")

    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    print("\nHippocampus test PASSED ✓")
