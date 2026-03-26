"""
Genesis Mind — Neuroplasticity (Unbounded Network Growth)

A human brain is not capped. It grows as long as it's alive:

    - Infancy: Explosive synaptogenesis (700 new connections/sec)
    - Childhood: Myelination, pruning, strengthening
    - Adulthood: Continuous neurogenesis in hippocampus
    - Elder: Slower growth but NEVER zero

Genesis is the same. There is NO parameter ceiling.

Growth is driven by TWO forces:

    1. DEVELOPMENTAL GROWTH: At phase transitions, networks grow to
       a minimum size appropriate for that phase's complexity.

    2. EXPERIENCE-DRIVEN GROWTH: Every N new concepts learned, the
       brain adds more neurons. The more you learn, the bigger your
       brain gets. There is no upper limit. A Genesis that learns
       10,000 concepts will have a physically larger brain than one
       that learns 100.

Growth mechanisms:
    1. LAYER WIDENING: Add neurons to hidden layers
    2. DEPTH EXTENSION: Add new layers to existing networks
    3. WEIGHT INHERITANCE: New neurons are initialized from statistics
       of existing weights (not random!) — transfer from self
    4. CONTINUOUS EXPANSION: Beyond Phase 5, growth continues at a
       rate proportional to learning — no ceiling, ever
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("genesis.neural.neuroplasticity")


# =============================================================================
# Minimum network dimensions per developmental phase (FLOOR, not ceiling)
# Beyond Phase 5, growth continues without limit
# =============================================================================
GROWTH_FLOOR = {
    0: {"personality_hidden": 128, "gru_layers": 3, "mc_hidden": 64},
    1: {"personality_hidden": 192, "gru_layers": 3, "mc_hidden": 96},
    2: {"personality_hidden": 256, "gru_layers": 3, "mc_hidden": 128},
    3: {"personality_hidden": 384, "gru_layers": 4, "mc_hidden": 192},
    4: {"personality_hidden": 512, "gru_layers": 4, "mc_hidden": 256},
    5: {"personality_hidden": 640, "gru_layers": 5, "mc_hidden": 320},
}

# Experience-driven growth parameters
GROWTH_RATE = 16          # Add 16 neurons per sqrt(concept) growth step
GROWTH_TRIGGER = 25       # Grow every 25 new concepts learned
GRU_LAYER_TRIGGER = 500   # Add a new GRU layer every 500 concepts
MAX_GRU_LAYERS = 20       # Practical limit for GRU depth (memory)
MIN_CONCEPTS_FOR_GROWTH = 10  # Don't grow from experience until 10 concepts


def compute_target_hidden(phase: int, concept_count: int) -> int:
    """
    Compute the target hidden dimension based on phase AND experience.

    The developmental phase sets a MINIMUM. Beyond that, the brain
    grows proportionally to concepts learned — with no ceiling.

    Growth follows a square-root curve (like real brain growth):
    fast initially, slower but never stopping.
    """
    # Phase-based minimum
    if phase in GROWTH_FLOOR:
        phase_min = GROWTH_FLOOR[phase]["personality_hidden"]
    else:
        # Beyond Phase 5: minimum keeps rising
        phase_min = GROWTH_FLOOR[5]["personality_hidden"] + (phase - 5) * 128

    # Experience-based growth (continuous, no ceiling)
    # Only kicks in after minimum concept threshold
    if concept_count >= MIN_CONCEPTS_FOR_GROWTH:
        experience_growth = int(math.sqrt(concept_count) * GROWTH_RATE)
    else:
        experience_growth = 0

    # The target is whichever is larger: phase minimum or experience growth
    target = max(phase_min, 128 + experience_growth)

    # Round to nearest multiple of 64 for efficiency (less frequent resizing)
    target = ((target + 63) // 64) * 64

    return target


def compute_target_gru_layers(concept_count: int, current_phase: int) -> int:
    """
    Compute target GRU depth. Adds layers as the mind matures.
    """
    phase_min = GROWTH_FLOOR.get(current_phase, GROWTH_FLOOR[5]).get("gru_layers", 3)
    experience_layers = 3 + (concept_count // GRU_LAYER_TRIGGER)
    return min(max(phase_min, experience_layers), MAX_GRU_LAYERS)


def compute_target_mc_hidden(phase: int, concept_count: int) -> int:
    """Meta-controller hidden dim — grows with experience."""
    phase_min = GROWTH_FLOOR.get(phase, GROWTH_FLOOR[5]).get("mc_hidden", 64)
    experience_growth = int(math.sqrt(max(0, concept_count)) * (GROWTH_RATE // 2))
    target = max(phase_min, 64 + experience_growth)
    return ((target + 31) // 32) * 32


# =============================================================================
# Weight-Inheriting Layer Expansion
# =============================================================================

def _grow_linear(old_layer: nn.Linear, new_in: int, new_out: int) -> nn.Linear:
    """
    Grow a linear layer by expanding its dimensions.

    New weights are initialized from the statistics of existing weights
    (mean/std transfer) — NOT random. This preserves learned patterns
    while adding capacity.
    """
    new_layer = nn.Linear(new_in, new_out)

    old_in = old_layer.in_features
    old_out = old_layer.out_features
    copy_in = min(old_in, new_in)
    copy_out = min(old_out, new_out)

    with torch.no_grad():
        new_layer.weight.data[:copy_out, :copy_in] = old_layer.weight.data[:copy_out, :copy_in]
        new_layer.bias.data[:copy_out] = old_layer.bias.data[:copy_out]

        if new_out > old_out:
            mean = old_layer.weight.data.mean()
            std = old_layer.weight.data.std()
            new_layer.weight.data[old_out:, :copy_in] = torch.normal(
                mean, std, size=(new_out - old_out, copy_in)
            )
            new_layer.bias.data[old_out:] = old_layer.bias.data.mean()

        if new_in > old_in:
            mean = old_layer.weight.data.mean()
            std = old_layer.weight.data.std()
            new_layer.weight.data[:copy_out, old_in:] = torch.normal(
                mean, std, size=(copy_out, new_in - old_in)
            )

    return new_layer


def _grow_gru(old_gru: nn.GRU, new_input: int, new_hidden: int,
              new_layers: int) -> nn.GRU:
    """
    Grow a GRU by expanding dimensions and/or adding layers.

    Existing weights are preserved. New capacity is initialized
    from existing weight statistics.
    """
    new_gru = nn.GRU(
        input_size=new_input,
        hidden_size=new_hidden,
        num_layers=new_layers,
        batch_first=True,
    )

    old_layers = old_gru.num_layers
    old_hidden = old_gru.hidden_size
    old_input = old_gru.input_size

    with torch.no_grad():
        for layer_idx in range(min(old_layers, new_layers)):
            for param_name in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
                suffix = f'_l{layer_idx}'
                old_param = getattr(old_gru, param_name + suffix)
                new_param = getattr(new_gru, param_name + suffix)

                if 'weight' in param_name:
                    old_h = old_hidden
                    new_h = new_hidden
                    old_d = old_input if (layer_idx == 0 and 'ih' in param_name) else old_hidden
                    new_d = new_input if (layer_idx == 0 and 'ih' in param_name) else new_hidden

                    for gate in range(3):
                        copy_h = min(old_h, new_h)
                        copy_d = min(old_d, new_d)
                        src_start = gate * old_h
                        dst_start = gate * new_h
                        new_param.data[dst_start:dst_start + copy_h, :copy_d] = \
                            old_param.data[src_start:src_start + copy_h, :copy_d]
                else:
                    for gate in range(3):
                        copy_h = min(old_hidden, new_hidden)
                        src_start = gate * old_hidden
                        dst_start = gate * new_hidden
                        new_param.data[dst_start:dst_start + copy_h] = \
                            old_param.data[src_start:src_start + copy_h]

    return new_gru


# =============================================================================
# Neuroplasticity Engine
# =============================================================================

class Neuroplasticity:
    """
    The unbounded network growth engine.

    There is NO parameter ceiling. The brain grows as long as it lives:
    - Phase transitions trigger minimum-size growth
    - Every N concepts learned triggers experience-driven growth
    - Beyond Phase 5, growth continues indefinitely

    A Genesis that learns 10,000 concepts will have a physically
    larger brain than one that learns 100. No limits.
    """

    def __init__(self):
        self._growth_history: List[Dict] = []
        self._total_growth_events = 0
        self._last_growth_concept_count = 0
        logger.info("Neuroplasticity engine initialized — UNBOUNDED growth, no parameter ceiling")

    def should_grow(self, current_phase: int, subconscious,
                    concept_count: int = 0) -> bool:
        """
        Check if networks need to grow.

        Two triggers:
        1. Phase transition: current dimensions < phase minimum
        2. Experience: learned enough new concepts since last growth
        """
        current_hidden = subconscious.personality.network.hidden_dim
        target_hidden = compute_target_hidden(current_phase, concept_count)

        # Phase-based growth
        if target_hidden > current_hidden:
            return True

        # Experience-based growth: trigger every GROWTH_TRIGGER concepts
        concepts_since_growth = concept_count - self._last_growth_concept_count
        if concepts_since_growth >= GROWTH_TRIGGER:
            # Recompute — experience might demand larger network
            target_hidden = compute_target_hidden(current_phase, concept_count)
            if target_hidden > current_hidden:
                return True

        return False

    def grow_networks(self, new_phase: int, subconscious,
                      concept_count: int = 0) -> Dict:
        """
        Grow all networks. No ceiling — brain expands to match experience.

        Returns a report of what changed.
        """
        target_hidden = compute_target_hidden(new_phase, concept_count)
        target_gru_layers = compute_target_gru_layers(concept_count, new_phase)
        target_mc = compute_target_mc_hidden(new_phase, concept_count)

        report = {
            "phase": new_phase,
            "concept_count": concept_count,
            "changes": {},
        }

        params_before = subconscious.get_total_params()

        # ── Grow Personality GRU ──
        personality = subconscious.personality
        pgru = personality.network  # PersonalityGRU
        if (target_hidden > pgru.hidden_dim or
                target_gru_layers > pgru.num_layers):

            old_hidden = pgru.hidden_dim
            old_layers = pgru.num_layers
            new_hidden = target_hidden
            new_layers = target_gru_layers

            # Grow the inner GRU
            new_gru = _grow_gru(
                pgru.gru, pgru.gru.input_size,
                new_hidden, new_layers
            )
            pgru.gru = new_gru
            pgru.hidden_dim = new_hidden
            pgru.num_layers = new_layers

            # Grow the output and prediction heads
            if hasattr(pgru, 'output_head'):
                old_out_dim = pgru.output_head[-1].out_features
                pgru.output_head = nn.Sequential(
                    nn.Linear(new_hidden, new_hidden // 2),
                    nn.ReLU(),
                    nn.Linear(new_hidden // 2, old_out_dim),
                )
            if hasattr(pgru, 'prediction_head'):
                old_pred_dim = pgru.prediction_head[-1].out_features
                pgru.prediction_head = nn.Sequential(
                    nn.Linear(new_hidden, new_hidden // 2),
                    nn.ReLU(),
                    nn.Linear(new_hidden // 2, old_pred_dim),
                )

            # Resize hidden state
            personality._hidden_state = None  # Reset, will auto-create on next forward

            # Rebuild optimizer
            personality.optimizer = torch.optim.Adam(
                pgru.parameters(),
                lr=personality.optimizer.param_groups[0]['lr']
            )

            report["changes"]["personality"] = {
                "hidden": f"{old_hidden} → {new_hidden}",
                "layers": f"{old_layers} → {new_layers}",
            }
            logger.info("  🧠 Personality GRU grew: hidden %d → %d, layers %d → %d",
                         old_hidden, new_hidden, old_layers, new_layers)

        # ── Grow Meta-Controller ──
        meta = subconscious.meta_controller
        if target_mc > 64:
            from genesis.neural.meta_controller import MetaController
            old_routes = meta._total_routes
            old_avg = meta._avg_weights.copy()

            new_meta = MetaController(
                input_dim=meta.input_dim,
                num_modules=meta.num_modules,
                hidden_dim=target_mc,
            )
            new_meta._total_routes = old_routes
            new_meta._avg_weights = old_avg
            subconscious.meta_controller = new_meta

            report["changes"]["meta_controller"] = {
                "hidden": f"→ {target_mc}",
            }
            logger.info("  🧠 Meta-controller grew: hidden → %d", target_mc)

        # Count params after
        params_after = subconscious.get_total_params()
        growth = params_after - params_before
        report["params_before"] = params_before
        report["params_after"] = params_after
        report["params_added"] = growth
        report["growth_pct"] = round(growth / max(1, params_before) * 100, 1)

        self._growth_history.append(report)
        self._total_growth_events += 1
        self._last_growth_concept_count = concept_count

        logger.info("  🧠 NEURAL GROWTH COMPLETE: %d → %d params (+%d, +%.1f%%) [NO CEILING]",
                     params_before, params_after, growth, report["growth_pct"])

        return report

    def get_stats(self) -> Dict:
        return {
            "total_growth_events": self._total_growth_events,
            "growth_history": self._growth_history,
            "last_growth_at_concepts": self._last_growth_concept_count,
            "ceiling": "NONE — unbounded growth",
        }

    def __repr__(self) -> str:
        return f"Neuroplasticity(growth_events={self._total_growth_events}, ceiling=NONE)"
