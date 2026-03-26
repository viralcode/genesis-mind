"""
Genesis Mind — Subconscious Orchestrator

This module ties together the entire neural cascade:

    Raw Input → Layer 1 (Encode + Instinct) → Layer 2 (Bind) → Layer 3 (Think)

It manages:
    - Weight persistence (save/load all networks as a single checkpoint)
    - The full forward pass through all three layers
    - Real-time training on every experience
    - Stats aggregation across all networks

The Subconscious is the "spine" of the Society of Mind.
All neural processing flows through it before reaching
the conscious reasoning engine (LLM).
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from genesis.neural.limbic_system import LimbicSystem
from genesis.neural.binding_network import BindingNetwork
from genesis.neural.personality_network import PersonalityNetwork
from genesis.neural.forward_model import WorldModel

logger = logging.getLogger("genesis.neural.subconscious")


class Subconscious:
    """
    The Society of Mind — all neural layers orchestrated.

    This is the complete subconscious pipeline sitting ON TOP of
    pre-trained evolutionary hardware (Whisper & CLIP):

        1. Evolutionary Hardware: CLIP (512-dim) & Whisper Text (384-dim)
        2. Limbic System:    embeddings  → neurochemical instinct
        3. Binding Network:  vis + aud   → 64-dim unified concept
        4. Personality Net:  concept + limbic + context → response

    All networks train in real-time on every experience.
    The saved weights collectively ARE the personality.
    """

    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Layer 1: Emotional evaluation of raw sensory data
        self.limbic_system = LimbicSystem(visual_dim=512, auditory_dim=384, lr=0.0005)

        # Layer 2: Associative Bridge
        self.binding_network = BindingNetwork(visual_dim=512, auditory_dim=384, output_dim=64, lr=0.001)

        # Layer 3: Conscious Executive
        self.personality = PersonalityNetwork(
            concept_dim=64, limbic_dim=4, context_dim=32,
            hidden_dim=128, lr=0.0005,
        )

        # Layer 4: World Model (Predictive Coding)
        self.world_model = WorldModel(concept_dim=64, hidden_dim=128, lr=0.001)

        # Load existing weights if they exist
        self._load_all()

        # Total params across all networks
        total_params = (
            sum(p.numel() for p in self.limbic_system.network.parameters()) +
            sum(p.numel() for p in self.binding_network.network.parameters()) +
            sum(p.numel() for p in self.personality.network.parameters()) +
            sum(p.numel() for p in self.world_model.network.parameters())
        )

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  SUBCONSCIOUS INITIALIZED — %d total parameters", total_params)
        logger.info("  Layer 1: Limbic Instinct (%d)", sum(p.numel() for p in self.limbic_system.network.parameters()))
        logger.info("  Layer 2: Binding (%d)", sum(p.numel() for p in self.binding_network.network.parameters()))
        logger.info("  Layer 3: Personality (%d)", sum(p.numel() for p in self.personality.network.parameters()))
        logger.info("  Layer 4: World Model (%d)", sum(p.numel() for p in self.world_model.network.parameters()))
        logger.info("  Using CLIP (512) and Text Embeddings (384) as pre-trained hardware.")
        logger.info("═══════════════════════════════════════════════════")

    def process_experience(self,
                           clip_embedding: Optional[np.ndarray] = None,
                           text_embedding: Optional[np.ndarray] = None,
                           context: Optional[np.ndarray] = None,
                           train: bool = True) -> Dict:
        """
        The full subconscious cascade on top of pre-trained hardware.

        Takes embeddings from CLIP and Whisper, and flows them through
        the plastic neural layers.

        Returns a dict with:
            - limbic_response: Dict of neurochemical levels
            - concept_embedding: 64-dim unified concept
            - personality_response: 64-dim response from the personality
            - consciousness_state: 128-dim current hidden state
        """
        result = {}

        visual_latent = clip_embedding if clip_embedding is not None else np.zeros(512, dtype=np.float32)
        auditory_latent = text_embedding if text_embedding is not None else np.zeros(384, dtype=np.float32)

        # ─── Layer 1: Encode (Instinct) ───────────────────────────
        limbic_response = self.limbic_system.react(visual_latent, auditory_latent)
        result['limbic_response'] = limbic_response

        # ─── Layer 2: Bind ────────────────────────────────────────
        concept_embedding = self.binding_network.bind(visual_latent, auditory_latent)
        result['concept_embedding'] = concept_embedding

        if train:
            # Replay buffer handles offline batch training in V3.1
            pass

        # ─── Layer 3: Think ───────────────────────────────────────
        response = self.personality.experience(
            concept_embedding=concept_embedding,
            limbic_state=limbic_response,
            context=context,
        )
        result['personality_response'] = response
        result['consciousness_state'] = self.personality.get_consciousness_state()
        
        if train:
            # Predict the future and learn from the surprise
            surprise = self.world_model.predict_and_learn(result['concept_embedding'], result['consciousness_state'])
            result['surprise'] = surprise

        return result

    def train_instinct(self, visual_features: Optional[np.ndarray],
                       auditory_features: Optional[np.ndarray],
                       target_chemicals: Dict[str, float]):
        """
        Train the limbic system with a supervised signal.

        Called after the conscious mind evaluates an experience:
        "This visual+audio pattern should make me feel X."
        Over time, the subconscious learns to react instinctively.
        """
        self.limbic_system.train_instinct(visual_features, auditory_features, target_chemicals)

    def consolidate_memories(self, replay_batch: list) -> float:
        """
        Train iteratively on a batch of past experiences (Dreaming/Sleep).
        Solves catastrophic forgetting by shuffling experiences.
        """
        if len(replay_batch) < 2:
            return 0.0
            
        v_list = [exp["visual"] for exp in replay_batch]
        a_list = [exp["auditory"] for exp in replay_batch]
        
        loss = self.binding_network.train_binding_batch(v_list, a_list)
        return loss

    def train_binding(self, visual_features: Optional[np.ndarray],
                      auditory_features: Optional[np.ndarray],
                      target_embedding: Optional[np.ndarray] = None):
        """Train the binding network with a supervised concept target."""
        self.binding_network.train_binding(visual_features, auditory_features, target_embedding)

    def save_all(self):
        """Save all neural weights — save the entire personality."""
        self.limbic_system.save_weights(self.weights_dir / "limbic_system.pt")
        self.binding_network.save_weights(self.weights_dir / "binding_network.pt")
        self.personality.save_weights(self.weights_dir / "personality.pt")
        self.world_model.save_weights(self.weights_dir / "world_model.pt")
        logger.info("All neural weights saved to %s", self.weights_dir)

    def _load_all(self):
        """Load all neural weights — restore the personality."""
        self.limbic_system.load_weights(self.weights_dir / "limbic_system.pt")
        self.binding_network.load_weights(self.weights_dir / "binding_network.pt")
        self.personality.load_weights(self.weights_dir / "personality.pt")
        self.world_model.load_weights(self.weights_dir / "world_model.pt")

    def get_stats(self) -> Dict:
        """Get comprehensive stats across all neural layers."""
        return {
            "layer_1": {
                "limbic_system": self.limbic_system.get_stats(),
            },
            "layer_2": {
                "binding_network": self.binding_network.get_stats(),
            },
            "layer_3": {
                "personality": self.personality.get_stats(),
            },
            "layer_4": {
                "world_model": self.world_model.get_stats(),
            },
        }

    def get_total_params(self) -> int:
        return (
            sum(p.numel() for p in self.limbic_system.network.parameters()) +
            sum(p.numel() for p in self.binding_network.network.parameters()) +
            sum(p.numel() for p in self.personality.network.parameters()) +
            sum(p.numel() for p in self.world_model.network.parameters())
        )

    def __repr__(self) -> str:
        return f"Subconscious(params={self.get_total_params()}, weights_dir={self.weights_dir})"
