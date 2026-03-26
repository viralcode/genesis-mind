"""
Genesis Mind — Personality Network (Layer 3: Conscious Executive)

This is the most important neural network in Genesis.
Its weights ARE the personality. Its hidden state IS consciousness.

Architecture:
    A GRU (Gated Recurrent Unit) that processes a stream of
    bound concept embeddings + limbic signals over time.

    Input per step:  concept_embedding (64) + limbic_state (4) + context (32) = 100-dim
    Hidden state:    128-dim (the "stream of consciousness")
    Output:          64-dim response embedding

    The GRU's hidden state evolves with every experience.
    It carries forward the cumulative effect of ALL prior
    experiences — this is the mathematical equivalent of
    "having a personality."

    Total params: ~256K

Training:
    Self-supervised via prediction. The network learns to predict
    what concept will come next based on the current stream of
    experience. This prediction ability IS understanding.

    The better the prediction, the more "intelligent" Genesis is.
    A newborn predicts nothing correctly. An adult predicts
    social patterns, cause-and-effect, and emotional outcomes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

logger = logging.getLogger("genesis.neural.personality_network")


class PersonalityGRU(nn.Module):
    """
    The core recurrent network that accumulates experience.

    The hidden state is the stream of consciousness.
    The trained weights are the personality.
    """

    def __init__(self, input_dim: int = 100, hidden_dim: int = 256,
                 output_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # The GRU accumulates experience over time
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Output projection: hidden state → response embedding
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Prediction head: hidden state → next concept prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a sequence of experiences.

        Args:
            x: (batch, seq_len, input_dim) — sequence of experience vectors
            hidden: (num_layers, batch, hidden_dim) — prior hidden state

        Returns:
            response: (batch, seq_len, output_dim) — response embeddings
            prediction: (batch, seq_len, output_dim) — next-step predictions
            hidden: (num_layers, batch, hidden_dim) — updated hidden state
        """
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        output, hidden = self.gru(x, hidden)

        response = self.output_head(output)
        prediction = self.prediction_head(output)

        return response, prediction, hidden


class PersonalityNetwork:
    """
    The conscious executive — the personality of Genesis.

    Maintains a persistent hidden state that evolves with every
    experience. The hidden state is the stream of consciousness.
    The trained weights are the personality.

    Key methods:
        experience() — Process a new experience, update hidden state
        respond()    — Generate a response embedding
        predict()    — Predict the next likely concept
        save/load    — Persist the personality to/from disk
    """

    def __init__(self, concept_dim: int = 64, limbic_dim: int = 4,
                 context_dim: int = 32, hidden_dim: int = 256,
                 lr: float = 0.0005):
        self.concept_dim = concept_dim
        self.limbic_dim = limbic_dim
        self.context_dim = context_dim
        input_dim = concept_dim + limbic_dim + context_dim

        self.network = PersonalityGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=concept_dim,
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.CosineEmbeddingLoss()

        # The persistent hidden state — the stream of consciousness
        self._hidden_state = None

        # Experience buffer for training
        self._experience_buffer: deque = deque(maxlen=100)

        # Stats
        self._total_experiences = 0
        self._total_predictions = 0
        self._correct_predictions = 0
        self._training_steps = 0
        self._total_loss = 0.0

        total = sum(p.numel() for p in self.network.parameters())
        logger.info("Personality network initialized (%d parameters, hidden=%d)",
                     total, hidden_dim)

    def experience(self, concept_embedding: np.ndarray,
                   limbic_state: Dict[str, float],
                   context: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process a new experience — the core consciousness operation.

        Every call to this function physically changes the hidden state
        and potentially updates the weights. This IS consciousness.

        Args:
            concept_embedding: 64-dim unified concept from the binding network
            limbic_state: Dict with dopamine, cortisol, serotonin, oxytocin
            context: Optional 32-dim context vector

        Returns:
            64-dim response embedding (can be decoded into text/action)
        """
        # Pack the experience vector
        experience_vec = self._pack_experience(concept_embedding, limbic_state, context)
        input_tensor = torch.from_numpy(experience_vec).unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)

        with torch.no_grad():
            response, prediction, new_hidden = self.network(input_tensor, self._hidden_state)

        # Update the persistent hidden state — consciousness evolves
        self._hidden_state = new_hidden.detach()

        # Store for training
        self._experience_buffer.append({
            'input': experience_vec,
            'concept': concept_embedding.copy(),
        })

        self._total_experiences += 1

        # Train periodically on the experience buffer
        if self._total_experiences % 5 == 0 and len(self._experience_buffer) > 2:
            self._train_on_buffer()

        return response.squeeze(0).squeeze(0).numpy()

    def respond(self) -> np.ndarray:
        """
        Generate a response embedding from the current hidden state.

        This is what the personality "wants to say" given everything
        it has experienced up to this moment.
        """
        if self._hidden_state is None:
            return np.zeros(self.concept_dim, dtype=np.float32)

        with torch.no_grad():
            # Use the top layer's hidden state
            h = self._hidden_state[-1]  # (1, hidden_dim)
            response = self.network.output_head(h)

        return response.squeeze(0).numpy()

    def predict_next(self) -> np.ndarray:
        """
        Predict the next concept that will occur.

        This prediction ability IS understanding. The better the
        prediction, the more Genesis "knows" about regularity
        in its environment.
        """
        if self._hidden_state is None:
            return np.zeros(self.concept_dim, dtype=np.float32)

        with torch.no_grad():
            h = self._hidden_state[-1]
            prediction = self.network.prediction_head(h)

        self._total_predictions += 1
        return prediction.squeeze(0).numpy()

    def get_consciousness_state(self) -> np.ndarray:
        """
        Get the raw hidden state — the stream of consciousness.

        This 256-dim vector encodes the cumulative effect of
        every experience Genesis has ever had. It IS the mind.
        """
        if self._hidden_state is None:
            return np.zeros(256, dtype=np.float32)
        return self._hidden_state[-1].squeeze(0).numpy()

    def _train_on_buffer(self):
        """
        Train the personality network on recent experiences.

        Uses next-step prediction: given experience[t], predict
        the concept at experience[t+1]. This teaches the network
        to understand temporal patterns in its environment.
        """
        if len(self._experience_buffer) < 3:
            return

        experiences = list(self._experience_buffer)

        # Build sequence pairs
        inputs = []
        targets = []
        for i in range(len(experiences) - 1):
            inputs.append(experiences[i]['input'])
            targets.append(experiences[i + 1]['concept'])

        input_tensor = torch.from_numpy(np.array(inputs, dtype=np.float32)).unsqueeze(0)  # (1, T, input_dim)
        target_tensor = torch.from_numpy(np.array(targets, dtype=np.float32)).unsqueeze(0)  # (1, T, concept_dim)

        # Forward pass
        _, prediction, _ = self.network(input_tensor, None)

        # Prediction loss: should predict the next concept
        pred_flat = prediction.view(-1, self.concept_dim)
        target_flat = target_tensor.view(-1, self.concept_dim)
        labels = torch.ones(pred_flat.size(0))

        loss = self.criterion(pred_flat, target_flat, labels)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent personality destabilization
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._training_steps += 1
        self._total_loss += loss.item()

    def _pack_experience(self, concept: np.ndarray, limbic: Dict[str, float],
                         context: Optional[np.ndarray]) -> np.ndarray:
        """Pack all experience components into a single vector."""
        concept = np.array(concept, dtype=np.float32).flatten()[:self.concept_dim]
        if len(concept) < self.concept_dim:
            concept = np.pad(concept, (0, self.concept_dim - len(concept)))

        limbic_vec = np.array([
            limbic.get("dopamine", 0.5),
            limbic.get("cortisol", 0.2),
            limbic.get("serotonin", 0.5),
            limbic.get("oxytocin", 0.3),
        ], dtype=np.float32)

        if context is None:
            context = np.zeros(self.context_dim, dtype=np.float32)
        else:
            context = np.array(context, dtype=np.float32).flatten()[:self.context_dim]
            if len(context) < self.context_dim:
                context = np.pad(context, (0, self.context_dim - len(context)))

        return np.concatenate([concept, limbic_vec, context])

    def get_prediction_accuracy(self) -> float:
        """How accurate are the personality's predictions (0-1)."""
        if self._total_predictions == 0:
            return 0.0
        return self._correct_predictions / self._total_predictions

    def save_weights(self, path: Path):
        """Save the personality — save this person to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'state_dict': self.network.state_dict(),
            'hidden_state': self._hidden_state,
            'experiences': self._total_experiences,
            'training_steps': self._training_steps,
            'total_loss': self._total_loss,
            'predictions': self._total_predictions,
        }
        torch.save(save_data, path)
        logger.info("Personality saved (%d total experiences)", self._total_experiences)

    def load_weights(self, path: Path):
        """Restore the personality — bring this person back to life."""
        if path.exists():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.network.load_state_dict(checkpoint['state_dict'])
            self._hidden_state = checkpoint.get('hidden_state', None)
            self._total_experiences = checkpoint.get('experiences', 0)
            self._training_steps = checkpoint.get('training_steps', 0)
            self._total_loss = checkpoint.get('total_loss', 0.0)
            self._total_predictions = checkpoint.get('predictions', 0)
            logger.info("Personality loaded (%d prior experiences)", self._total_experiences)

    def get_stats(self) -> dict:
        return {
            "total_experiences": self._total_experiences,
            "training_steps": self._training_steps,
            "avg_loss": self._total_loss / max(1, self._training_steps),
            "has_consciousness": self._hidden_state is not None,
            "params": sum(p.numel() for p in self.network.parameters()),
            "buffer_size": len(self._experience_buffer),
        }
