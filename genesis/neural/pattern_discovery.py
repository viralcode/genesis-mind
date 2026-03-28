"""
Genesis Mind — Acoustic Pattern Discovery

Statistical word discovery from raw VQ token streams.
This is how human infants learn word boundaries — NOT from explicit
teaching, but from transitional probability patterns in heard speech.

Reference: Saffran, Aslin & Newport (1996) — "Statistical Learning
by 8-Month-Old Infants"

The key insight: within a word, syllable transitions have HIGH
probability (e.g., ba→by is very likely in "baby"). Between words,
transitions have LOW probability (e.g., "the baby" → "the" follows
a low-probability transition from "by" to "the").

We apply the same principle to VQ token sequences:
1. Track bigram frequencies of VQ tokens
2. Detect "drops" in transitional probability = word boundaries
3. Frequent segments between boundaries = word-like units
4. Store frequently recurring segments as acoustic exemplars
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("genesis.neural.pattern_discovery")


class FrequentPatternDetector:
    """
    Discovers recurring acoustic patterns from VQ token streams.
    
    Operates in two phases:
    
    Phase 1 (Accumulation): Count n-gram frequencies from all heard
    VQ tokens. Track which token sequences recur most often.
    
    Phase 2 (Discovery): When a token pattern has been heard N+ times,
    it's likely a real acoustic unit (word, syllable, or phoneme group).
    Report it for storage in AcousticWordMemory.
    
    This runs automatically in the brain daemon's auditory tick —
    no 'teach' command needed.
    """

    def __init__(self,
                 min_pattern_length: int = 3,
                 max_pattern_length: int = 15,
                 discovery_threshold: int = 5,
                 max_patterns: int = 200):
        """
        Args:
            min_pattern_length: Minimum VQ tokens for a valid pattern
            max_pattern_length: Maximum VQ tokens for a valid pattern
            discovery_threshold: How many times a pattern must recur
                                 before it becomes a "word"
            max_patterns: Maximum number of discovered patterns
        """
        self.min_len = min_pattern_length
        self.max_len = max_pattern_length
        self.discovery_threshold = discovery_threshold
        self.max_patterns = max_patterns

        # Token bigram counts for transitional probability
        self._bigram_counts: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._unigram_counts: Dict[int, int] = defaultdict(int)
        self._total_tokens_seen = 0

        # Pattern frequency tracking
        # key = tuple of token IDs, value = count
        self._pattern_counts: Dict[tuple, int] = defaultdict(int)

        # Already-discovered patterns (avoid re-reporting)
        self._discovered: set = set()

        # Total tokens processed
        self._chunks_processed = 0

        logger.info(
            "Pattern discovery initialized (threshold=%d, length=%d-%d)",
            discovery_threshold, min_pattern_length, max_pattern_length,
        )

    def observe(self, vq_tokens: List[int]) -> List[Tuple[str, List[int]]]:
        """
        Observe a chunk of VQ tokens from heard audio.
        
        Updates bigram statistics and checks for recurring patterns.
        
        Returns:
            List of (suggested_name, token_pattern) for newly discovered
            patterns that exceeded the recurrence threshold.
        """
        if not vq_tokens or len(vq_tokens) < self.min_len:
            return []

        self._chunks_processed += 1

        # Update unigram and bigram counts
        for token in vq_tokens:
            self._unigram_counts[token] += 1
            self._total_tokens_seen += 1

        for i in range(len(vq_tokens) - 1):
            self._bigram_counts[vq_tokens[i]][vq_tokens[i + 1]] += 1

        # Segment the token stream using transitional probability drops
        segments = self._segment_by_tp(vq_tokens)

        # Count pattern occurrences
        discoveries = []
        for segment in segments:
            if len(segment) < self.min_len or len(segment) > self.max_len:
                continue

            # Check diversity — reject monotone sequences
            if len(set(segment)) < 2:
                continue

            key = tuple(segment)
            self._pattern_counts[key] += 1

            # Check if this pattern just crossed the discovery threshold
            if (self._pattern_counts[key] >= self.discovery_threshold
                    and key not in self._discovered
                    and len(self._discovered) < self.max_patterns):

                self._discovered.add(key)
                name = f"acoustic_{len(self._discovered)}"
                discoveries.append((name, list(key)))
                logger.info(
                    "🔊 Discovered pattern '%s': %s (heard %d times)",
                    name, list(key)[:8], self._pattern_counts[key],
                )

        return discoveries

    def _segment_by_tp(self, tokens: List[int]) -> List[List[int]]:
        """
        Segment a token sequence using transitional probability drops.
        
        A "word boundary" is placed where the transitional probability
        P(token_n+1 | token_n) drops significantly compared to the
        running average.
        
        This is the core algorithm from Saffran et al. — infants
        use exactly this mechanism to discover word boundaries in
        continuous speech.
        """
        if len(tokens) < 2:
            return [tokens]

        # Compute transitional probabilities for each position
        tps = []
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            count_bigram = self._bigram_counts.get(t1, {}).get(t2, 0)
            count_unigram = self._unigram_counts.get(t1, 1)
            tp = count_bigram / max(count_unigram, 1)
            tps.append(tp)

        if not tps:
            return [tokens]

        # Find TP drops — positions where TP falls below the mean
        # These are likely word boundaries
        mean_tp = sum(tps) / len(tps) if tps else 0.5
        threshold = mean_tp * 0.5  # Boundary when TP drops to 50% of mean

        segments = []
        current_segment = [tokens[0]]

        for i, tp in enumerate(tps):
            if tp < threshold and len(current_segment) >= self.min_len:
                # Word boundary detected
                segments.append(current_segment)
                current_segment = [tokens[i + 1]]
            else:
                current_segment.append(tokens[i + 1])

        if current_segment:
            segments.append(current_segment)

        # Also extract fixed-length sliding windows as candidates
        # (catches patterns that TP segmentation misses)
        for window_size in range(self.min_len, min(self.max_len + 1, len(tokens) + 1)):
            step = max(1, window_size // 2)
            for start in range(0, len(tokens) - window_size + 1, step):
                segment = tokens[start:start + window_size]
                if len(set(segment)) >= 2:  # Diversity check
                    segments.append(segment)

        return segments

    def get_stats(self) -> dict:
        return {
            "chunks_processed": self._chunks_processed,
            "total_tokens_seen": self._total_tokens_seen,
            "unique_tokens_seen": len(self._unigram_counts),
            "patterns_tracked": len(self._pattern_counts),
            "patterns_discovered": len(self._discovered),
            "top_patterns": self._get_top_patterns(5),
        }

    def _get_top_patterns(self, n: int = 5) -> List[dict]:
        """Get the N most frequent patterns."""
        sorted_patterns = sorted(
            self._pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]
        return [
            {"tokens": list(k), "count": v, "discovered": k in self._discovered}
            for k, v in sorted_patterns
        ]

    def __repr__(self) -> str:
        return (
            f"FrequentPatternDetector("
            f"discovered={len(self._discovered)}, "
            f"tracked={len(self._pattern_counts)}, "
            f"tokens_seen={self._total_tokens_seen})"
        )
