"""
Genesis Mind — Grammar Acquisition (Dual Mode)

Language is not innate. A child does not emerge from the womb
speaking perfect English. Grammar is LEARNED through exposure,
repetition, and pattern recognition.

This module provides two modes of language generation:

    Mode A: LLM-Assisted (default)
        Uses phi3:mini with phase-constrained prompts.
        The LLM provides fluent grammar but it is "borrowed"
        from pre-training — not truly learned from scratch.

    Mode B: Tabula Rasa (pure learning)
        A lightweight N-gram language model trained ONLY from
        speech heard through the microphone. Starts completely
        mute. Learns word sequences purely by statistical
        frequency of what the Creator says.

        Newborn:   "..."
        Infant:    "apple" (single words)
        Toddler:   "me want apple" (bigrams)
        Child:     "can I have the apple please" (trigrams+)

The grammar mode can be switched at runtime.
"""

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("genesis.cortex.grammar")


class NgramLanguageModel:
    """
    A pure Tabula-Rasa language model.

    Learns grammar EXCLUSIVELY from transcribed speech heard
    through the microphone. No pre-trained weights. No internet
    knowledge. Just raw statistical patterns from the Creator's voice.

    Internally maintains:
    - Unigram counts (word frequencies)
    - Bigram counts (word-pair frequencies)
    - Trigram counts (word-triplet frequencies)

    Generation works by:
    1. Pick a start word (weighted by frequency)
    2. For each next word, sample from the conditional distribution
       P(w_n | w_{n-1}, w_{n-2}) using available n-gram data
    3. Stop at max length or end-of-sentence token
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._unigrams: Dict[str, int] = defaultdict(int)
        self._bigrams: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._trigrams: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total_words_heard = 0
        self._total_sentences_heard = 0
        self._storage_path = storage_path
        self._load()

        logger.info("N-gram language model initialized (%d words heard, %d sentences)",
                     self._total_words_heard, self._total_sentences_heard)

    def learn_from_speech(self, text: str):
        """
        Learn grammar patterns from a sentence heard from the Creator.

        This is the ONLY training mechanism. The model has no other
        source of linguistic knowledge.
        """
        words = text.lower().strip().split()
        if not words:
            return

        self._total_sentences_heard += 1

        # Learn unigrams
        for w in words:
            self._unigrams[w] += 1
            self._total_words_heard += 1

        # Learn bigrams
        for i in range(len(words) - 1):
            self._bigrams[words[i]][words[i + 1]] += 1

        # Learn trigrams
        for i in range(len(words) - 2):
            key = f"{words[i]}|{words[i + 1]}"
            self._trigrams[key][words[i + 2]] += 1

        # Periodically save
        if self._total_sentences_heard % 10 == 0:
            self._save()

    def generate(self, max_words: int = 10, temperature: float = 1.0) -> str:
        """
        Generate a sentence using learned n-gram statistics.

        If the model has heard very little, responses will be
        primitive and fragmented — exactly like a real child.
        """
        vocab_size = len(self._unigrams)

        # Phase 0: No data at all → silence
        if vocab_size == 0:
            return "..."

        # Phase 1: Very few words → single word responses
        if vocab_size < 5:
            word = self._sample_unigram(temperature)
            return word if word else "..."

        # Phase 2: Some bigrams → two-word combinations
        if self._total_sentences_heard < 10:
            w1 = self._sample_unigram(temperature)
            w2 = self._sample_bigram(w1, temperature)
            return f"{w1} {w2}" if w2 else w1

        # Phase 3+: Full n-gram generation
        sentence = []
        current_word = self._sample_unigram(temperature)
        sentence.append(current_word)

        for _ in range(max_words - 1):
            # Try trigram first
            if len(sentence) >= 2:
                key = f"{sentence[-2]}|{sentence[-1]}"
                next_word = self._sample_trigram(key, temperature)
                if next_word:
                    sentence.append(next_word)
                    continue

            # Fall back to bigram
            next_word = self._sample_bigram(current_word, temperature)
            if next_word:
                sentence.append(next_word)
                current_word = next_word
            else:
                break  # No continuation found

        return " ".join(sentence)

    def _sample_unigram(self, temperature: float = 1.0) -> str:
        """Sample a word based on frequency."""
        if not self._unigrams:
            return ""
        words = list(self._unigrams.keys())
        counts = list(self._unigrams.values())
        probs = self._apply_temperature(counts, temperature)
        return random.choices(words, weights=probs, k=1)[0]

    def _sample_bigram(self, prev_word: str, temperature: float = 1.0) -> Optional[str]:
        """Sample the next word given the previous word."""
        if prev_word not in self._bigrams:
            return None
        next_words = self._bigrams[prev_word]
        if not next_words:
            return None
        words = list(next_words.keys())
        counts = list(next_words.values())
        probs = self._apply_temperature(counts, temperature)
        return random.choices(words, weights=probs, k=1)[0]

    def _sample_trigram(self, key: str, temperature: float = 1.0) -> Optional[str]:
        """Sample the next word given the previous two words."""
        if key not in self._trigrams:
            return None
        next_words = self._trigrams[key]
        if not next_words:
            return None
        words = list(next_words.keys())
        counts = list(next_words.values())
        probs = self._apply_temperature(counts, temperature)
        return random.choices(words, weights=probs, k=1)[0]

    @staticmethod
    def _apply_temperature(counts: List[int], temperature: float) -> List[float]:
        """Apply temperature scaling to probability distribution."""
        if temperature <= 0:
            temperature = 0.01
        log_probs = [c ** (1.0 / temperature) for c in counts]
        total = sum(log_probs)
        return [p / total for p in log_probs] if total > 0 else [1.0 / len(counts)] * len(counts)

    def get_vocab_size(self) -> int:
        return len(self._unigrams)

    def get_stats(self) -> Dict:
        return {
            "vocab_size": len(self._unigrams),
            "total_words_heard": self._total_words_heard,
            "total_sentences_heard": self._total_sentences_heard,
            "bigram_pairs": sum(len(v) for v in self._bigrams.values()),
            "trigram_pairs": sum(len(v) for v in self._trigrams.values()),
        }

    def _save(self):
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "unigrams": dict(self._unigrams),
            "bigrams": {k: dict(v) for k, v in self._bigrams.items()},
            "trigrams": {k: dict(v) for k, v in self._trigrams.items()},
            "total_words": self._total_words_heard,
            "total_sentences": self._total_sentences_heard,
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
            self._unigrams = defaultdict(int, data.get("unigrams", {}))
            raw_bigrams = data.get("bigrams", {})
            for k, v in raw_bigrams.items():
                self._bigrams[k] = defaultdict(int, v)
            raw_trigrams = data.get("trigrams", {})
            for k, v in raw_trigrams.items():
                self._trigrams[k] = defaultdict(int, v)
            self._total_words_heard = data.get("total_words", 0)
            self._total_sentences_heard = data.get("total_sentences", 0)
            logger.info("Loaded n-gram model: %d words, %d sentences",
                         self._total_words_heard, self._total_sentences_heard)
        except Exception as e:
            logger.error("Failed to load n-gram model: %s", e)


class GrammarEngine:
    """
    Dual-mode grammar system.

    Switches between LLM-assisted and pure Tabula-Rasa n-gram
    generation based on configuration.
    """

    def __init__(self, mode: str = "llm",
                 ngram_storage_path: Optional[Path] = None):
        self._mode = mode
        self._ngram_model = NgramLanguageModel(storage_path=ngram_storage_path)

        logger.info("Grammar engine initialized (mode=%s)", mode)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in ("llm", "tabula_rasa"):
            raise ValueError(f"Invalid grammar mode: {value}. Use 'llm' or 'tabula_rasa'.")
        self._mode = value
        logger.info("Grammar mode changed to: %s", value)

    def learn_from_speech(self, text: str):
        """
        Feed heard speech to the n-gram model.

        This is always called regardless of mode — the n-gram model
        is always learning in the background. The mode only affects
        which model is used for GENERATION.
        """
        self._ngram_model.learn_from_speech(text)

    def generate_response(self, context: str = "", reasoning_engine=None,
                          identity: str = "", moral_context: str = "",
                          phase: int = 0, phase_name: str = "Newborn",
                          memories: list = None) -> str:
        """
        Generate a response using the active grammar mode.

        In LLM mode: delegates to the ReasoningEngine.
        In Tabula Rasa mode: uses the n-gram model exclusively.
        """
        if self._mode == "tabula_rasa":
            # Pure n-gram generation — no pretrained knowledge
            max_words = min(3 + phase * 3, 30)
            response = self._ngram_model.generate(max_words=max_words, temperature=0.8)
            logger.info("Tabula Rasa response: '%s'", response)
            return response

        # LLM mode — delegate to reasoning engine
        if reasoning_engine is not None:
            thought = reasoning_engine.think(
                question=context,
                memories=memories or [],
                identity=identity,
                moral_context=moral_context,
                phase=phase,
                phase_name=phase_name,
            )
            return thought.content

        # Fallback if no reasoning engine
        return self._ngram_model.generate(max_words=10)

    def get_ngram_stats(self) -> Dict:
        return self._ngram_model.get_stats()

    def __repr__(self) -> str:
        return (
            f"GrammarEngine(mode={self._mode}, "
            f"ngram_vocab={self._ngram_model.get_vocab_size()})"
        )
