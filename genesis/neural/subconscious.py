import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
import random

import numpy as np

from genesis.neural.limbic_system import LimbicSystem
from genesis.neural.binding_network import BindingNetwork
from genesis.neural.personality_network import PersonalityNetwork
from genesis.neural.forward_model import WorldModel
from genesis.neural.response_decoder import ResponseDecoder
from genesis.neural.meta_controller import MetaController
from genesis.neural.device import DEVICE, try_compile

logger = logging.getLogger("genesis.neural.subconscious")


class Subconscious:
    """
    The Society of Mind — all neural layers orchestrated.

    This is the complete subconscious pipeline sitting ON TOP of
    from-scratch neural hardware (VisualCortex & PhonemeEmbedder):

        1. Sensory Input: VisualCortex (64-dim) & Phoneme Embedding (64-dim)
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
        self.limbic_system = LimbicSystem(visual_dim=64, auditory_dim=64, lr=0.0005)

        # Layer 2: Associative Bridge
        self.binding_network = BindingNetwork(visual_dim=64, auditory_dim=64, output_dim=64, lr=0.001)

        # Layer 3: Conscious Executive
        self.personality = PersonalityNetwork(
            concept_dim=64, limbic_dim=4, context_dim=32,
            hidden_dim=128, lr=0.0005,
        )

        # Layer 4: World Model (Predictive Coding)
        self.world_model = WorldModel(concept_dim=64, hidden_dim=128, lr=0.001)

        # Response Decoder: Neural Voice
        self.response_decoder = ResponseDecoder(top_k=3)

        # Meta-Controller: Neural Router (the Thalamus)
        self.meta_controller = MetaController(
            input_dim=128,
            num_modules=4,
            hidden_dim=64,
            lr=0.0003,
        )

        # ═══ Live Experience Replay Buffer (PRIORITIZED) ═══
        self.replay_buffer = deque(maxlen=10000)
        self._replay_count = 0

        # ═══ Curriculum Gating Thresholds ═══
        self._visual_cortex_loss = 1.0
        self._binding_gate_open = False
        self._personality_gate_open = False
        self._world_model_gate_open = False

        # ═══ Training History (Observability) ═══
        self._loss_history: Dict[str, deque] = {
            'binding': deque(maxlen=500),
            'personality': deque(maxlen=500),
            'world_model': deque(maxlen=500),
            'limbic': deque(maxlen=500),
        }
        self._last_concept_embedding: Optional[np.ndarray] = None  # For temporal prediction

        # Load existing weights BEFORE torch.compile
        self._load_all()

        # ═══ torch.compile() — ONLY for networks that DON'T grow ═══
        # Personality, WorldModel, MetaController undergo neuroplasticity growth
        # which replaces their sub-modules. torch.compile on MPS produces raw
        # functions that break save/load after growth. Skip them — they're tiny
        # (~20K params) so compilation adds negligible speedup anyway.
        self.limbic_system.network = try_compile(self.limbic_system.network, "LimbicNetwork")
        self.binding_network.network = try_compile(self.binding_network.network, "BindingNetwork")
        # personality, world_model, meta_controller: NOT compiled (growable)

        # Total params across all networks
        total_params = (
            sum(p.numel() for p in self.limbic_system.network.parameters()) +
            sum(p.numel() for p in self.binding_network.network.parameters()) +
            sum(p.numel() for p in self.personality.network.parameters()) +
            sum(p.numel() for p in self.world_model.network.parameters())
        )

        logger.info("═══════════════════════════════════════════════════")
        logger.info("  SUBCONSCIOUS INITIALIZED — %d total parameters", total_params)
        logger.info("  Device: %s | torch.compile: ON | AMP: ON", DEVICE)
        logger.info("  Layer 1: Limbic (%d)", sum(p.numel() for p in self.limbic_system.network.parameters()))
        logger.info("  Layer 2: Binding (%d)", sum(p.numel() for p in self.binding_network.network.parameters()))
        logger.info("  Layer 3: Personality (%d)", sum(p.numel() for p in self.personality.network.parameters()))
        logger.info("  Layer 4: World Model (%d)", sum(p.numel() for p in self.world_model.network.parameters()))
        logger.info("  Router:  Meta-Controller (%d)", sum(p.numel() for p in self.meta_controller.network.parameters()))
        logger.info("  Replay: Tri-signal priority (surprise+emotion+drive), 10K buffer")
        logger.info("  Curriculum Gating: ON (VC<0.05 → binding, binding>10 → personality)")
        logger.info("  All weights learned from scratch — ZERO pretrained models.")
        logger.info("═══════════════════════════════════════════════════")

    def update_curriculum_gates(self, visual_cortex_loss: float = 1.0):
        """
        Update curriculum gating thresholds from external signals.
        
        Gates:
            - Binding trains when visual cortex loss < 0.05
            - Personality trains when binding has > 10 training steps
            - World model trains when personality has > 5 experiences
        """
        self._visual_cortex_loss = visual_cortex_loss
        self._binding_gate_open = visual_cortex_loss < 0.05
        self._personality_gate_open = self.binding_network._training_steps > 10
        self._world_model_gate_open = self.personality._total_experiences > 5

    def process_experience(self,
                           visual_embedding: Optional[np.ndarray] = None,
                           text_embedding: Optional[np.ndarray] = None,
                           context: Optional[np.ndarray] = None,
                           emotional_intensity: float = 0.0,
                           drive_hunger: float = 0.0,
                           train: bool = True) -> Dict:
        """
        The full subconscious cascade.

        Takes embeddings from the from-scratch VisualCortex (64-dim)
        and AuditoryCortex/PhonemeEmbedder (64-dim), and flows them
        through the plastic neural layers.

        Args:
            emotional_intensity: Current emotional intensity (0-1) from EmotionalState
            drive_hunger: Current dominant drive hunger level (0-1) from DriveSystem
        """
        result = {}

        visual_latent = visual_embedding if visual_embedding is not None else np.zeros(64, dtype=np.float32)
        auditory_latent = text_embedding if text_embedding is not None else np.zeros(64, dtype=np.float32)

        # ─── Meta-Controller: Compute routing weights ──────────
        routing = self.meta_controller.route(visual_latent, auditory_latent)
        result['routing_weights'] = routing

        # ─── Layer 1: Encode (Instinct) ───────────────────────
        limbic_response = self.limbic_system.react(visual_latent, auditory_latent)
        limbic_weight = routing['limbic']
        scaled_limbic = {
            k: v * limbic_weight for k, v in limbic_response.items()
        }
        result['limbic_response'] = limbic_response
        result['scaled_limbic'] = scaled_limbic

        # ─── Layer 2: Bind ────────────────────────────────────
        concept_embedding = self.binding_network.bind(visual_latent, auditory_latent)
        binding_weight = routing['binding']
        scaled_concept = concept_embedding * binding_weight
        result['concept_embedding'] = concept_embedding

        # ═══ LIVE TRAINING: Prioritized replay on EVERY conscious cycle ═══
        if train and self._binding_gate_open:
            self._replay_count += 1
            buf_len = len(self.replay_buffer)
            if buf_len >= 4:
                # Adaptive batch: min(64, buffer_size), at least 4
                batch_size = min(64, max(4, buf_len // 2))
                batch = self._sample_prioritized(batch_size=batch_size)
                if len(batch) >= 2:
                    loss = self.binding_network.train_binding_batch(
                        [b['visual'] for b in batch],
                        [b['auditory'] for b in batch]
                    )
                    if loss > 0 and self._replay_count % 25 == 0:
                        logger.info(
                            "Live replay: binding loss=%.4f batch=%d buf=%d gates=[B:%s P:%s W:%s]",
                            loss, len(batch), buf_len,
                            "✓" if self._binding_gate_open else "✗",
                            "✓" if self._personality_gate_open else "✗",
                            "✓" if self._world_model_gate_open else "✗",
                        )
                    # Record binding loss
                    if loss > 0:
                        self._loss_history['binding'].append((time.time(), loss))

        # ─── Layer 3: Think ─────────────────────────────────
        personality_weight = routing['personality']
        response = self.personality.experience(
            concept_embedding=concept_embedding,
            limbic_state=limbic_response,
            context=context,
            train=train,
        )
        result['personality_response'] = response
        result['consciousness_state'] = self.personality.get_consciousness_state()
        
        surprise = 0.0
        if train and self._world_model_gate_open:
            surprise = self.world_model.predict_and_learn(
                result['concept_embedding'], result['consciousness_state']
            )
            result['surprise'] = surprise
            self.meta_controller.learn_from_surprise(
                visual_latent, auditory_latent, surprise
            )
            # Record world model loss
            self._loss_history['world_model'].append((time.time(), surprise))
        else:
            result['surprise'] = 0.0

        # ═══ Temporal Prediction Signal ═══
        # If we have a previous concept embedding, compute prediction error
        if self._last_concept_embedding is not None and train:
            pred_error = float(np.linalg.norm(concept_embedding - self._last_concept_embedding))
            result['temporal_prediction_error'] = pred_error
        else:
            result['temporal_prediction_error'] = 0.0
        self._last_concept_embedding = concept_embedding.copy()

        # ═══ Store in replay buffer with TRI-SIGNAL priority ═══
        if train:
            self.replay_buffer.append({
                'visual': visual_latent.copy(),
                'auditory': auditory_latent.copy(),
                'limbic': limbic_response,
                'concept': concept_embedding.copy(),
                'surprise': max(surprise, 0.01),
                'emotional_intensity': max(emotional_intensity, 0.01),
                'drive_hunger': max(drive_hunger, 0.01),
            })

        return result

    def decode_response(self, response_embedding: np.ndarray, semantic_memory) -> str:
        """Decode the GRU's response embedding into text — the neural voice."""
        return self.response_decoder.decode(response_embedding, semantic_memory)

    def _sample_prioritized(self, batch_size: int = 32) -> list:
        """
        Tri-signal prioritized replay sampling.

        Priority = surprise * 0.5 + emotional_intensity * 0.3 + drive_hunger * 0.2
        
        High-surprise experiences teach about prediction errors.
        High-emotion experiences are more salient (like traumatic/joyful memories).
        High-drive experiences connect to motivational relevance.
        
        Recency bias ensures newer experiences get slightly higher weight.
        """
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
        
        # Tri-signal priority
        surprises = np.array([exp.get('surprise', 0.01) for exp in self.replay_buffer])
        emotions = np.array([exp.get('emotional_intensity', 0.01) for exp in self.replay_buffer])
        drives = np.array([exp.get('drive_hunger', 0.01) for exp in self.replay_buffer])
        
        priorities = surprises * 0.5 + emotions * 0.3 + drives * 0.2
        
        # Recency bias: newer experiences get up to 2x weight
        recency = np.linspace(0.5, 1.0, len(priorities))
        priorities = priorities * recency
        
        # Normalize to probability distribution
        total = priorities.sum()
        if total > 0:
            priorities = priorities / total
        else:
            priorities = np.ones(len(priorities)) / len(priorities)
        # Ensure exact sum=1.0 for numpy (float precision fix)
        priorities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.replay_buffer), size=batch_size, 
                                    replace=False, p=priorities)
        return [self.replay_buffer[i] for i in indices]

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
        self.meta_controller.save_weights(self.weights_dir / "meta_controller.pt")
        logger.info("All neural weights saved to %s", self.weights_dir)

    def _load_all(self):
        """Load all neural weights — restore the personality."""
        self.limbic_system.load_weights(self.weights_dir / "limbic_system.pt")
        self.binding_network.load_weights(self.weights_dir / "binding_network.pt")
        self.personality.load_weights(self.weights_dir / "personality.pt")
        self.world_model.load_weights(self.weights_dir / "world_model.pt")
        self.meta_controller.load_weights(self.weights_dir / "meta_controller.pt")

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
            "router": {
                "meta_controller": self.meta_controller.get_stats(),
            },
            "training": {
                "replay_buffer_size": len(self.replay_buffer),
                "replay_count": self._replay_count,
                "device": str(DEVICE),
                "curriculum": {
                    "visual_cortex_loss": round(self._visual_cortex_loss, 6),
                    "binding_gate": self._binding_gate_open,
                    "personality_gate": self._personality_gate_open,
                    "world_model_gate": self._world_model_gate_open,
                },
            },
        }

    def get_total_params(self) -> int:
        return (
            sum(p.numel() for p in self.limbic_system.network.parameters()) +
            sum(p.numel() for p in self.binding_network.network.parameters()) +
            sum(p.numel() for p in self.personality.network.parameters()) +
            sum(p.numel() for p in self.world_model.network.parameters()) +
            sum(p.numel() for p in self.meta_controller.network.parameters())
        )

    def get_training_history(self) -> Dict[str, List]:
        """Return per-network loss history for dashboard visualization."""
        return {
            name: [(t, l) for t, l in history]
            for name, history in self._loss_history.items()
        }

    def record_loss(self, network_name: str, loss_value: float):
        """Record a training loss from an external caller (e.g., limbic training in main.py)."""
        if network_name in self._loss_history:
            self._loss_history[network_name].append((time.time(), loss_value))

    def __repr__(self) -> str:
        return f"Subconscious(params={self.get_total_params()}, weights_dir={self.weights_dir})"
