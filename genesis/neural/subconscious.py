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

from genesis.neural.visual_cortex import VisualCortex
from genesis.neural.auditory_cortex import AuditoryCortex
from genesis.neural.limbic_system import LimbicSystem
from genesis.neural.binding_network import BindingNetwork
from genesis.neural.personality_network import PersonalityNetwork

logger = logging.getLogger("genesis.neural.subconscious")


class Subconscious:
    """
    The Society of Mind — all neural layers orchestrated.

    This is the complete subconscious pipeline:

        1. Visual Cortex:    raw frame   → 64-dim visual latent
        2. Auditory Cortex:  raw audio   → 32-dim auditory latent
        3. Limbic System:    latents     → neurochemical instinct
        4. Binding Network:  vis + aud   → 64-dim unified concept
        5. Personality Net:  concept + limbic + context → response

    All networks train in real-time on every experience.
    The saved weights collectively ARE the personality.
    """

    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Layer 1: Subconscious Sentinels
        self.visual_cortex = VisualCortex(latent_dim=64, lr=0.001)
        self.auditory_cortex = AuditoryCortex(n_mels=64, latent_dim=32, lr=0.001)
        self.limbic_system = LimbicSystem(visual_dim=64, auditory_dim=32, lr=0.0005)

        # Layer 2: Associative Bridge
        self.binding_network = BindingNetwork(visual_dim=64, auditory_dim=32, output_dim=64, lr=0.001)

        # Layer 3: Conscious Executive
        self.personality = PersonalityNetwork(
            concept_dim=64, limbic_dim=4, context_dim=32,
            hidden_dim=128, lr=0.0005,
        )

        # Load existing weights if they exist
        self._load_all()

        # Total params across all networks
        total_params = (
            sum(p.numel() for p in self.visual_cortex.parameters()) +
            sum(p.numel() for p in self.auditory_cortex.parameters()) +
            sum(p.numel() for p in self.limbic_system.network.parameters()) +
            sum(p.numel() for p in self.binding_network.network.parameters()) +
            sum(p.numel() for p in self.personality.network.parameters())
        )

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  SUBCONSCIOUS INITIALIZED — %d total parameters", total_params)
        logger.info("  Layer 1: Visual (%d) + Auditory (%d) + Limbic (%d)",
                     sum(p.numel() for p in self.visual_cortex.parameters()),
                     sum(p.numel() for p in self.auditory_cortex.parameters()),
                     sum(p.numel() for p in self.limbic_system.network.parameters()))
        logger.info("  Layer 2: Binding (%d)",
                     sum(p.numel() for p in self.binding_network.network.parameters()))
        logger.info("  Layer 3: Personality (%d)",
                     sum(p.numel() for p in self.personality.network.parameters()))
        logger.info("═══════════════════════════════════════════════════")

    def process_experience(self,
                           visual_frame: Optional[np.ndarray] = None,
                           audio_chunk: Optional[np.ndarray] = None,
                           audio_sr: int = 16000,
                           context: Optional[np.ndarray] = None,
                           train: bool = True) -> Dict:
        """
        The full subconscious cascade.

        Takes raw sensory input and flows it through all layers,
        training each one along the way.

        Returns a dict with:
            - visual_latent: 64-dim visual features
            - auditory_latent: 32-dim auditory features
            - limbic_response: Dict of neurochemical levels
            - concept_embedding: 64-dim unified concept
            - personality_response: 64-dim response from the personality
            - consciousness_state: 128-dim current hidden state
        """
        result = {}

        # ─── Layer 1: Encode ──────────────────────────────────────
        # Visual
        if visual_frame is not None:
            if train:
                loss = self.visual_cortex.train_on_frame(visual_frame)
                result['visual_loss'] = loss
            visual_latent = self.visual_cortex.encode(visual_frame)
        else:
            visual_latent = np.zeros(64, dtype=np.float32)
        result['visual_latent'] = visual_latent

        # Auditory
        if audio_chunk is not None and len(audio_chunk) > 0:
            if train:
                loss = self.auditory_cortex.train_on_audio(audio_chunk, audio_sr)
                result['auditory_loss'] = loss
            auditory_latent = self.auditory_cortex.encode(audio_chunk, audio_sr)
        else:
            auditory_latent = np.zeros(32, dtype=np.float32)
        result['auditory_latent'] = auditory_latent

        # Limbic (instinct)
        limbic_response = self.limbic_system.react(visual_latent, auditory_latent)
        result['limbic_response'] = limbic_response

        # ─── Layer 2: Bind ────────────────────────────────────────
        concept_embedding = self.binding_network.bind(visual_latent, auditory_latent)
        result['concept_embedding'] = concept_embedding

        if train:
            self.binding_network.train_binding(visual_latent, auditory_latent)

        # ─── Layer 3: Think ───────────────────────────────────────
        response = self.personality.experience(
            concept_embedding=concept_embedding,
            limbic_state=limbic_response,
            context=context,
        )
        result['personality_response'] = response
        result['consciousness_state'] = self.personality.get_consciousness_state()

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

    def train_binding(self, visual_features: Optional[np.ndarray],
                      auditory_features: Optional[np.ndarray],
                      target_embedding: Optional[np.ndarray] = None):
        """Train the binding network with a supervised concept target."""
        self.binding_network.train_binding(visual_features, auditory_features, target_embedding)

    def save_all(self):
        """Save all neural weights — save the entire personality."""
        self.visual_cortex.save_weights(self.weights_dir / "visual_cortex.pt")
        self.auditory_cortex.save_weights(self.weights_dir / "auditory_cortex.pt")
        self.limbic_system.save_weights(self.weights_dir / "limbic_system.pt")
        self.binding_network.save_weights(self.weights_dir / "binding_network.pt")
        self.personality.save_weights(self.weights_dir / "personality.pt")
        logger.info("All neural weights saved to %s", self.weights_dir)

    def _load_all(self):
        """Load all neural weights — restore the personality."""
        self.visual_cortex.load_weights(self.weights_dir / "visual_cortex.pt")
        self.auditory_cortex.load_weights(self.weights_dir / "auditory_cortex.pt")
        self.limbic_system.load_weights(self.weights_dir / "limbic_system.pt")
        self.binding_network.load_weights(self.weights_dir / "binding_network.pt")
        self.personality.load_weights(self.weights_dir / "personality.pt")

    def get_stats(self) -> Dict:
        """Get comprehensive stats across all neural layers."""
        return {
            "layer_1": {
                "visual_cortex": self.visual_cortex.get_stats(),
                "auditory_cortex": self.auditory_cortex.get_stats(),
                "limbic_system": self.limbic_system.get_stats(),
            },
            "layer_2": {
                "binding_network": self.binding_network.get_stats(),
            },
            "layer_3": {
                "personality": self.personality.get_stats(),
            },
        }

    def get_total_params(self) -> int:
        return (
            sum(p.numel() for p in self.visual_cortex.parameters()) +
            sum(p.numel() for p in self.auditory_cortex.parameters()) +
            sum(p.numel() for p in self.limbic_system.network.parameters()) +
            sum(p.numel() for p in self.binding_network.network.parameters()) +
            sum(p.numel() for p in self.personality.network.parameters())
        )

    def __repr__(self) -> str:
        return f"Subconscious(params={self.get_total_params()}, weights_dir={self.weights_dir})"
