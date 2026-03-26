"""
Genesis Mind — Intrinsic Curiosity Engine

A human child does not wait to be taught. It actively explores,
points at unknown objects, and asks "What's that?" relentlessly.

This module gives Genesis a curiosity drive by measuring the
**novelty** (surprise) of every perception. When something
unfamiliar is detected, Genesis spontaneously asks the Creator.

Mathematical Basis:
    Surprise(x) = 1 - max_{c ∈ Memory} cos(z_x, z_c)

    If Surprise > threshold → "What is that?"
    If Surprise ≈ 0       → Familiar, no action needed.

Curiosity is also subject to **habituation**: repeated exposure
to the same novel stimulus reduces the curiosity response, just
as a child eventually stops asking about the same thing.
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict, deque

logger = logging.getLogger("genesis.cortex.curiosity")


@dataclass
class CuriosityEvent:
    """A record of Genesis being curious about something."""
    stimulus: str              # What triggered curiosity
    surprise_score: float      # How novel it was (0-1)
    question_asked: str        # What Genesis asked
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    answered: bool = False     # Whether the Creator responded


class CuriosityEngine:
    """
    The intrinsic motivation system.

    Measures novelty in perceptions and generates curiosity-driven
    questions when encountering unfamiliar stimuli.
    """

    def __init__(self, surprise_threshold: float = 0.7,
                 habituation_rate: float = 0.15,
                 curiosity_cooldown_sec: float = 15.0,
                 max_history: int = 200):
        self._surprise_threshold = surprise_threshold
        self._habituation_rate = habituation_rate
        self._cooldown_sec = curiosity_cooldown_sec

        # Habituation map: stimulus_hash → exposure count
        self._exposure_counts: Dict[str, int] = defaultdict(int)

        # Recent curiosity events
        self._history: List[CuriosityEvent] = []
        self._max_history = max_history

        # Unanswered question queue — questions asked but never resolved
        self._unanswered: deque = deque(maxlen=50)

        # Cooldown tracking
        self._last_question_time: float = 0.0

        # Stats
        self._total_questions = 0
        self._total_evaluations = 0

        logger.info("Curiosity engine initialized (threshold=%.2f)", surprise_threshold)

    def compute_surprise(self, embedding: np.ndarray,
                         known_embeddings: List[np.ndarray]) -> float:
        """
        Compute how surprising/novel a perception is.

        Surprise = 1 - max similarity to any known concept.
        A perception identical to something known has surprise ≈ 0.
        A completely novel perception has surprise ≈ 1.

        Args:
            embedding: The embedding vector of the new perception
            known_embeddings: List of all known concept embeddings

        Returns:
            Surprise score between 0.0 and 1.0
        """
        if not known_embeddings or embedding is None:
            return 1.0  # Everything is novel when you know nothing

        embedding = np.array(embedding).flatten()
        max_similarity = 0.0

        for known in known_embeddings:
            known = np.array(known).flatten()
            # Cosine similarity
            dot = np.dot(embedding, known)
            norm_prod = np.linalg.norm(embedding) * np.linalg.norm(known)
            if norm_prod > 0:
                sim = dot / norm_prod
                max_similarity = max(max_similarity, sim)

        surprise = 1.0 - max_similarity
        self._total_evaluations += 1
        return float(np.clip(surprise, 0.0, 1.0))

    def should_ask(self, surprise: float, stimulus_key: str = "") -> bool:
        """
        Determine if Genesis should ask a curiosity question.

        Factors:
        1. Surprise must exceed the threshold
        2. Must not be in cooldown period (avoid spamming questions)
        3. Habituation: repeated exposure reduces curiosity
        """
        # Check cooldown
        now = time.time()
        if now - self._last_question_time < self._cooldown_sec:
            return False

        # Apply habituation: reduce effective surprise based on exposure
        if stimulus_key:
            exposure = self._exposure_counts[stimulus_key]
            habituation_factor = max(0.0, 1.0 - exposure * self._habituation_rate)
            effective_surprise = surprise * habituation_factor
            self._exposure_counts[stimulus_key] += 1
        else:
            effective_surprise = surprise

        return effective_surprise > self._surprise_threshold

    def generate_question(self, context: str = "", phase: int = 0) -> str:
        """
        Generate a curiosity-driven question appropriate to the
        current developmental phase.
        """
        self._last_question_time = time.time()
        self._total_questions += 1

        if phase <= 1:
            # Infant: simply points (minimal language)
            question = "...? (looks at something new)"
        elif phase == 2:
            # Toddler: "What that?"
            question = f"What that?" if not context else f"What is '{context}'?"
        elif phase == 3:
            # Child: proper questions
            question = f"Creator, what is '{context}'? I haven't seen this before."
        elif phase == 4:
            # Adolescent: deeper curiosity
            question = f"I notice something unfamiliar: '{context}'. Can you explain what it is and why it matters?"
        else:
            # Adult: sophisticated curiosity
            question = f"I've encountered something novel: '{context}'. I'd like to understand its nature and how it relates to what I already know."

        event = CuriosityEvent(
            stimulus=context,
            surprise_score=1.0,
            question_asked=question,
        )
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Track this as unanswered until explicitly resolved
        self._unanswered.append(event)

        return question

    def get_stats(self) -> Dict:
        """Get curiosity statistics."""
        return {
            "total_questions_asked": self._total_questions,
            "total_evaluations": self._total_evaluations,
            "unique_stimuli_encountered": len(self._exposure_counts),
            "cooldown_sec": self._cooldown_sec,
            "surprise_threshold": self._surprise_threshold,
        }

    def mark_answered(self, stimulus_key: str):
        """Mark a curiosity question as answered (the Creator responded)."""
        for event in self._unanswered:
            if event.stimulus == stimulus_key and not event.answered:
                event.answered = True
                break
        # Remove answered events from the queue
        self._unanswered = deque(
            [e for e in self._unanswered if not e.answered],
            maxlen=50,
        )

    def get_unanswered(self) -> List[CuriosityEvent]:
        """Return all unanswered curiosity questions."""
        return list(self._unanswered)

    def get_most_burning_question(self) -> Optional[str]:
        """
        Return the most urgent unanswered question.

        Priority: highest surprise score among unanswered questions.
        Returns None if all questions have been answered.
        """
        if not self._unanswered:
            return None
        most_burning = max(self._unanswered, key=lambda e: e.surprise_score)
        return most_burning.question_asked

    def reset_habituation(self):
        """Reset habituation (e.g., after sleep — the child is fresh again)."""
        self._exposure_counts.clear()
        logger.info("Habituation reset — curiosity restored")

    def __repr__(self) -> str:
        return (
            f"CuriosityEngine(questions={self._total_questions}, "
            f"threshold={self._surprise_threshold})"
        )
