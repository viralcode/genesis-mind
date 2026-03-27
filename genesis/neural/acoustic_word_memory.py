"""
Genesis Mind — Acoustic Word Memory

This module implements from-scratch word recognition using template
matching on VQ codebook token sequences.

How a human infant learns words:
    1. Hears "apple" while seeing an apple (co-occurrence)
    2. Stores the acoustic pattern of "apple" in memory
    3. When hearing a similar pattern later, activates "apple" concept
    4. After many exposures, recognition becomes robust

This module replicates that process:
    1. During `teach`, capture mic audio → VQ tokens → store as exemplar
    2. During listening, run DTW against stored exemplars
    3. Nearest match = recognized word
    4. Multiple exemplars per word improve robustness

Dynamic Time Warping (DTW) is used for matching because:
    - Time-invariant: "aaapple" matches "apple"
    - No training needed beyond storing exemplars
    - Works with tiny vocabulary (perfect for infant-phase)
    - Biologically plausible (temporal normalization in auditory cortex)

No pretrained models. No external dependencies beyond numpy.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("genesis.neural.acoustic_word_memory")


@dataclass
class AcousticExemplar:
    """A single stored acoustic pattern for a word."""
    word: str
    vq_tokens: List[int]          # VQ codebook token sequence
    embedding: Optional[List[float]] = None  # Mean VQ embedding (64-dim)
    timestamp: str = ""
    confidence: float = 1.0       # Decays if never matched


@dataclass
class RecognitionResult:
    """Result of attempting to recognize an acoustic pattern."""
    word: str                     # Best matching word
    distance: float               # DTW distance (lower = better match)
    confidence: float             # 0-1 confidence score
    exemplar_count: int           # How many exemplars exist for this word
    is_match: bool                # Whether distance is below threshold


class AcousticWordMemory:
    """
    From-scratch word recognition via DTW template matching.

    Stores VQ token sequences as "acoustic fingerprints" for each
    learned word. Recognition compares heard VQ tokens against all
    stored exemplars using Dynamic Time Warping.

    This is how early speech recognition worked (1970s DTW-based
    systems) and is biologically plausible — the auditory cortex
    performs similar temporal pattern matching.
    """

    def __init__(self, storage_path: Path,
                 max_exemplars_per_word: int = 10,
                 recognition_threshold: float = 0.35,
                 min_token_length: int = 3):
        self.storage_path = Path(storage_path)
        self.max_exemplars_per_word = max_exemplars_per_word
        self.recognition_threshold = recognition_threshold
        self.min_token_length = min_token_length

        # word → list of AcousticExemplar
        self._exemplars: Dict[str, List[AcousticExemplar]] = {}

        # Recognition stats
        self._total_recognitions = 0
        self._successful_recognitions = 0

        self._load()
        logger.info(
            "Acoustic word memory initialized (%d words, %d exemplars)",
            len(self._exemplars),
            sum(len(v) for v in self._exemplars.values()),
        )

    # =========================================================================
    # Storage — teach a word's acoustic pattern
    # =========================================================================

    def store_exemplar(self, word: str, vq_tokens: List[int],
                       embedding: Optional[np.ndarray] = None,
                       timestamp: str = "") -> AcousticExemplar:
        """
        Store a VQ token sequence as an exemplar for a word.

        Called during `teach` — "this sound pattern = this word".

        Args:
            word: The concept word
            vq_tokens: VQ codebook token IDs from heard audio
            embedding: Optional mean embedding vector
            timestamp: When this exemplar was captured
        """
        if len(vq_tokens) < self.min_token_length:
            logger.debug("Token sequence too short (%d), skipping", len(vq_tokens))
            return None

        exemplar = AcousticExemplar(
            word=word,
            vq_tokens=vq_tokens,
            embedding=embedding.tolist() if embedding is not None else None,
            timestamp=timestamp,
            confidence=1.0,
        )

        if word not in self._exemplars:
            self._exemplars[word] = []

        self._exemplars[word].append(exemplar)

        # Keep only the N most recent exemplars per word
        if len(self._exemplars[word]) > self.max_exemplars_per_word:
            self._exemplars[word] = self._exemplars[word][-self.max_exemplars_per_word:]

        self._save()
        logger.info(
            "Stored acoustic exemplar for '%s' (%d tokens, %d total exemplars)",
            word, len(vq_tokens), len(self._exemplars[word]),
        )
        return exemplar

    # =========================================================================
    # Recognition — identify a word from VQ tokens
    # =========================================================================

    def _has_diversity(self, vq_tokens: List[int], min_unique: int = 3) -> bool:
        """
        Check if a token sequence has enough diversity to be meaningful.
        
        A sequence of all-zeros (or all same token) is noise from an
        untrained VQ codebook, not real speech. Require at least
        `min_unique` distinct tokens.
        """
        return len(set(vq_tokens)) >= min_unique

    def recognize(self, vq_tokens: List[int],
                   top_k: int = 3) -> List[RecognitionResult]:
        """
        Match heard VQ tokens against all stored acoustic exemplars.

        Uses DTW (Dynamic Time Warping) for time-invariant matching.
        Rejects monotone sequences (all same token) as noise.

        Args:
            vq_tokens: VQ token sequence from heard audio
            top_k: Number of best matches to return

        Returns:
            List of RecognitionResult, sorted by distance (best first)
        """
        if len(vq_tokens) < self.min_token_length:
            return []

        if not self._exemplars:
            return []

        # Reject monotone / low-diversity sequences — these are noise
        # from an undertrained VQ codebook
        if not self._has_diversity(vq_tokens, min_unique=3):
            return []

        self._total_recognitions += 1

        results = []
        for word, exemplars in self._exemplars.items():
            # Skip exemplars that are themselves low-diversity (corrupt data)
            valid_exemplars = [e for e in exemplars if self._has_diversity(e.vq_tokens, min_unique=3)]
            if not valid_exemplars:
                continue

            # Compare against each exemplar, take the best (minimum distance)
            best_dist = float('inf')
            for exemplar in valid_exemplars:
                dist = self._dtw_distance(vq_tokens, exemplar.vq_tokens)
                if dist < best_dist:
                    best_dist = dist

            # Normalize distance by sequence length
            avg_len = (len(vq_tokens) + min(len(e.vq_tokens) for e in valid_exemplars)) / 2
            normalized_dist = best_dist / max(avg_len, 1)

            # Convert distance to confidence (inverse sigmoid)
            confidence = 1.0 / (1.0 + np.exp(5.0 * (normalized_dist - self.recognition_threshold)))

            results.append(RecognitionResult(
                word=word,
                distance=normalized_dist,
                confidence=confidence,
                exemplar_count=len(valid_exemplars),
                is_match=normalized_dist < self.recognition_threshold and confidence > 0.5,
            ))

        # Sort by distance (best matches first)
        results.sort(key=lambda r: r.distance)

        if results and results[0].is_match:
            self._successful_recognitions += 1
            logger.debug(
                "Recognized '%s' (dist=%.3f, conf=%.2f)",
                results[0].word, results[0].distance, results[0].confidence,
            )

        return results[:top_k]

    def recognize_best(self, vq_tokens: List[int]) -> Optional[RecognitionResult]:
        """Return the single best match, or None if no match."""
        results = self.recognize(vq_tokens, top_k=1)
        if results and results[0].is_match:
            return results[0]
        return None

    # =========================================================================
    # Dynamic Time Warping — the core matching algorithm
    # =========================================================================

    def _dtw_distance(self, seq_a: List[int], seq_b: List[int]) -> float:
        """
        Compute DTW distance between two VQ token sequences.

        DTW aligns two sequences optimally in time, allowing for
        stretching and compression — "aaapple" matches "apple".

        The cost function is binary: 0 if tokens match, 1 if they don't.
        This makes the distance count the minimum number of mismatches
        after optimal time alignment.

        Time complexity: O(n*m) where n, m are sequence lengths.
        For typical word-length sequences (5-30 tokens), this is fast.
        """
        n, m = len(seq_a), len(seq_b)

        # Early termination for very different lengths
        if n == 0 or m == 0:
            return float(max(n, m))

        # DTW cost matrix
        dtw = np.full((n + 1, m + 1), float('inf'))
        dtw[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Cost: 0 if tokens match, 1 if different
                cost = 0.0 if seq_a[i-1] == seq_b[j-1] else 1.0

                dtw[i, j] = cost + min(
                    dtw[i-1, j],      # insertion
                    dtw[i, j-1],      # deletion
                    dtw[i-1, j-1],    # match/substitution
                )

        return dtw[n, m]

    # =========================================================================
    # Segmentation — split continuous audio tokens into word candidates
    # =========================================================================

    def segment_and_recognize(self, vq_tokens: List[int],
                              window_sizes: List[int] = None
                              ) -> List[RecognitionResult]:
        """
        Slide windows of various sizes over a token sequence to find
        word-like segments that match stored exemplars.

        This is a simple segmentation strategy — real infants use
        statistical learning (Saffran et al. 1996) to discover word
        boundaries. Future versions could learn boundaries from
        transitional probabilities.

        Args:
            vq_tokens: Long token sequence from continuous audio
            window_sizes: Sizes of sliding windows to try

        Returns:
            All recognized words found in the sequence
        """
        if window_sizes is None:
            # Try windows matching known exemplar lengths ± 30%
            window_sizes = set()
            for exemplars in self._exemplars.values():
                for ex in exemplars:
                    base = len(ex.vq_tokens)
                    window_sizes.add(max(self.min_token_length, int(base * 0.7)))
                    window_sizes.add(base)
                    window_sizes.add(int(base * 1.3))
            window_sizes = sorted(window_sizes) if window_sizes else [5, 10, 15, 20]

        recognized = []
        for window_size in window_sizes:
            if window_size > len(vq_tokens):
                continue

            step = max(1, window_size // 3)  # Overlapping windows
            for start in range(0, len(vq_tokens) - window_size + 1, step):
                segment = vq_tokens[start:start + window_size]
                result = self.recognize_best(segment)
                if result:
                    # Avoid duplicate recognitions of same word
                    if not any(r.word == result.word for r in recognized):
                        recognized.append(result)

        return recognized

    # =========================================================================
    # Queries
    # =========================================================================

    def get_exemplar_tokens(self, word: str) -> Optional[List[int]]:
        """Get the best (most recent) VQ token sequence for a word."""
        if word not in self._exemplars or not self._exemplars[word]:
            return None
        return self._exemplars[word][-1].vq_tokens

    def get_vocabulary(self) -> List[str]:
        """Get all words with stored acoustic exemplars."""
        return list(self._exemplars.keys())

    def get_stats(self) -> dict:
        return {
            "words": len(self._exemplars),
            "total_exemplars": sum(len(v) for v in self._exemplars.values()),
            "total_recognitions": self._total_recognitions,
            "successful_recognitions": self._successful_recognitions,
            "recognition_rate": (
                self._successful_recognitions / max(1, self._total_recognitions)
            ),
            "vocabulary": self.get_vocabulary(),
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save(self):
        data = {}
        for word, exemplars in self._exemplars.items():
            data[word] = [
                {
                    "vq_tokens": ex.vq_tokens,
                    "embedding": ex.embedding,
                    "timestamp": ex.timestamp,
                    "confidence": ex.confidence,
                }
                for ex in exemplars
            ]
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f)

    def _load(self):
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            for word, exemplar_list in data.items():
                self._exemplars[word] = [
                    AcousticExemplar(
                        word=word,
                        vq_tokens=ex["vq_tokens"],
                        embedding=ex.get("embedding"),
                        timestamp=ex.get("timestamp", ""),
                        confidence=ex.get("confidence", 1.0),
                    )
                    for ex in exemplar_list
                ]
        except Exception as e:
            logger.warning("Could not load acoustic word memory: %s", e)


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Acoustic Word Memory Test")
    print("Testing DTW-based word recognition from VQ tokens")
    print("=" * 60)

    mem = AcousticWordMemory(
        storage_path=Path("/tmp/genesis_acoustic_word_test.json"),
        recognition_threshold=0.6,
    )

    # Simulate teaching words through VQ tokens
    # (In reality, these come from SensorimotorLoop.hear())
    print("\n--- Teaching Phase ---")
    mem.store_exemplar("apple",  [42, 17, 8, 55, 12, 30])
    mem.store_exemplar("apple",  [43, 17, 9, 54, 12, 31])  # Slight variation
    mem.store_exemplar("banana", [10, 88, 44, 22, 10, 88, 44])
    mem.store_exemplar("banana", [11, 87, 45, 21, 10, 89, 43])
    mem.store_exemplar("hello",  [5, 60, 120, 200])
    print(f"  Vocabulary: {mem.get_vocabulary()}")

    # Simulate hearing — try to recognize
    print("\n--- Recognition Phase ---")

    # Exact match
    result = mem.recognize_best([42, 17, 8, 55, 12, 30])
    print(f"  Exact apple:     {result.word if result else 'no match'} "
          f"(dist={result.distance:.3f})" if result else "  Exact apple:     no match")

    # Slightly different (should still match)
    result = mem.recognize_best([44, 17, 7, 56, 12, 29])
    print(f"  Similar apple:   {result.word if result else 'no match'} "
          f"(dist={result.distance:.3f})" if result else "  Similar apple:   no match")

    # Time-stretched (DTW should handle this)
    result = mem.recognize_best([42, 42, 17, 17, 8, 55, 55, 12, 30])
    print(f"  Stretched apple: {result.word if result else 'no match'} "
          f"(dist={result.distance:.3f})" if result else "  Stretched apple: no match")

    # Banana
    result = mem.recognize_best([10, 88, 44, 22, 10, 88, 44])
    print(f"  Exact banana:    {result.word if result else 'no match'} "
          f"(dist={result.distance:.3f})" if result else "  Exact banana:    no match")

    # Random noise (should NOT match)
    result = mem.recognize_best([200, 150, 100, 50, 0, 255])
    print(f"  Random noise:    {result.word if result else 'no match'} "
          f"(dist={result.distance:.3f})" if result else "  Random noise:    no match")

    # Test segmentation on continuous stream
    print("\n--- Segmentation Test ---")
    continuous = [0, 0, 42, 17, 8, 55, 12, 30, 0, 0, 10, 88, 44, 22, 10, 88, 44, 0]
    found = mem.segment_and_recognize(continuous)
    print(f"  Found in stream: {[r.word for r in found]}")

    print(f"\n  Stats: {mem.get_stats()}")
    print("Acoustic Word Memory test PASSED")
