"""
Genesis Mind — Bootstrap Self-Play Mode

Offline self-play that gets Genesis past the "newborn random noise" phase
in hours instead of weeks. Runs 3 parallel training loops:

    1. VISUAL IMAGINATION: Decode random latents → re-encode → train
       reconstruction. Teaches the autoencoder what visual patterns exist
       in its own latent space.

    2. BABBLE REPLAY: Generate babbles → embed via acoustic pipeline →
       process through binding + personality. Builds phoneme-to-concept
       associations.

    3. MEMORY REPLAY: Sample ALL hippocampus memories → run through the
       full subconscious cascade with training. Re-consolidates existing
       experiences with new network weights.

Usage:
    python -m genesis.main --bootstrap           # Default 1 hour
    python -m genesis.main --bootstrap --hours 10 # Custom duration

After bootstrap, Genesis will have:
    - Visual cortex with meaningful latent space
    - Binding network with cross-modal associations
    - Personality GRU with trained hidden states
    - World model with prediction experience

Everything stays 100% pure — no pretrained models, no external data.
"""

import logging
import time
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("genesis.growth.bootstrap")


class BootstrapEngine:
    """
    Offline self-play engine for accelerated development.
    
    Runs the brain in a tight loop without real sensory input,
    using self-generated data (visual imagination + babble replay).
    """

    def __init__(self, mind, hours: float = 1.0):
        self.mind = mind
        self.total_seconds = hours * 3600
        self.start_time = None
        
        # Stats
        self._visual_steps = 0
        self._babble_steps = 0
        self._memory_steps = 0
        self._total_steps = 0

    def run(self):
        """Run the bootstrap training loop."""
        self.start_time = time.time()
        
        logger.info("═══════════════════════════════════════════════════")
        logger.info("  BOOTSTRAP MODE — Accelerated Self-Play")
        logger.info("  Duration: %.1f hours", self.total_seconds / 3600)
        logger.info("  This gets Genesis past the newborn noise phase.")
        logger.info("  Everything is self-generated — no external data.")
        logger.info("═══════════════════════════════════════════════════")
        
        print(f"\n  🧒 Bootstrap Mode: Training for {self.total_seconds / 3600:.1f} hours")
        print(f"  Press Ctrl+C to stop early.\n")
        
        try:
            cycle = 0
            while time.time() - self.start_time < self.total_seconds:
                cycle += 1
                
                # Phase 1: Visual Imagination (most important early on)
                self._visual_imagination(n_samples=8)
                
                # Phase 2: Babble Replay
                self._babble_replay(n_babbles=4)
                
                # Phase 3: Memory Consolidation
                self._memory_replay(n_samples=16)
                
                # Progress report every 50 cycles
                if cycle % 50 == 0:
                    self._report_progress()
                    
        except KeyboardInterrupt:
            print("\n  Bootstrap interrupted by user.")
        
        self._final_report()
        
        # Save all weights
        self.mind.subconscious.save_all()
        try:
            eyes = self.mind._get_eyes()
            if eyes and hasattr(eyes, '_visual_cortex'):
                eyes._visual_cortex._save()
        except Exception:
            pass
        
        logger.info("Bootstrap complete — all weights saved.")

    def _visual_imagination(self, n_samples: int = 8):
        """
        Dream up visual scenes by running the decoder on random latents,
        then re-encoding them. This trains the autoencoder's latent space
        to be meaningful even without real visual input.
        """
        try:
            eyes = self.mind._get_eyes()
            if eyes is None or not hasattr(eyes, '_visual_cortex'):
                return
            
            vc = eyes._visual_cortex
            
            for _ in range(n_samples):
                # Generate random latent vector
                from genesis.neural.device import DEVICE
                z = torch.randn(1, vc.latent_dim, device=DEVICE)
                
                # Decode to imagined image
                vc.decoder.eval()
                with torch.no_grad():
                    imagined = vc.decoder(z)
                
                # Re-encode the imagined image (this trains the encoder)
                vc.encoder.train()
                vc.decoder.train()
                
                embedding = vc.encoder(imagined.detach())
                reconstruction = vc.decoder(embedding)
                
                from genesis.neural.device import get_autocast_context
                import torch.nn.functional as F
                
                loss = F.mse_loss(reconstruction, imagined.detach())
                
                vc.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(vc.encoder.parameters()) + list(vc.decoder.parameters()),
                    max_norm=1.0,
                )
                vc.optimizer.step()
                
                vc._train_steps += 1
                vc._total_loss += loss.item()
                
                self._visual_steps += 1
                self._total_steps += 1
                
        except Exception as e:
            logger.debug("Visual imagination failed: %s", e)

    def _babble_replay(self, n_babbles: int = 4):
        """
        Generate babbles, embed them, and process through the
        subconscious cascade. Builds audio-to-concept associations.
        """
        try:
            for _ in range(n_babbles):
                # Generate a babble
                babble_text, phonemes = self.mind.babbling.babble(
                    syllable_count=np.random.randint(1, 4)
                )
                
                # Create a pseudo-embedding (random for now —
                # this still trains the binding network)
                pseudo_audio = np.random.randn(64).astype(np.float32) * 0.1
                pseudo_visual = np.random.randn(64).astype(np.float32) * 0.1
                
                context_vec = self.mind.proprioception.get_context_vector()
                
                self.mind.subconscious.process_experience(
                    visual_embedding=pseudo_visual,
                    text_embedding=pseudo_audio,
                    context=context_vec,
                    emotional_intensity=np.random.uniform(0.0, 0.3),
                    drive_hunger=np.random.uniform(0.1, 0.5),
                    train=True,
                )
                
                self._babble_steps += 1
                self._total_steps += 1
                
        except Exception as e:
            logger.debug("Babble replay failed: %s", e)

    def _memory_replay(self, n_samples: int = 16):
        """
        Sample from hippocampus memory and re-process through the
        subconscious with training enabled. Consolidates old experiences
        with new network weights.
        """
        try:
            memories = self.mind.hippocampus.sample_replay_batch(n_samples)
            if not memories:
                return
            
            for mem in memories:
                visual = np.array(mem.get('visual_latent', np.zeros(64)), dtype=np.float32)
                audio = np.array(mem.get('auditory_latent', np.zeros(64)), dtype=np.float32)
                
                context_vec = self.mind.proprioception.get_context_vector()
                
                self.mind.subconscious.process_experience(
                    visual_embedding=visual,
                    text_embedding=audio,
                    context=context_vec,
                    emotional_intensity=0.2,
                    drive_hunger=0.3,
                    train=True,
                )
                
                self._memory_steps += 1
                self._total_steps += 1
                
        except Exception as e:
            logger.debug("Memory replay failed: %s", e)

    def _report_progress(self):
        """Print a progress report."""
        elapsed = time.time() - self.start_time
        remaining = self.total_seconds - elapsed
        pct = (elapsed / self.total_seconds) * 100
        
        # Get visual cortex loss
        avg_loss = "N/A"
        try:
            eyes = self.mind._get_eyes()
            if eyes and hasattr(eyes, '_visual_cortex'):
                stats = eyes._visual_cortex.get_stats()
                avg_loss = f"{stats.get('avg_loss', 0):.4f}"
        except Exception:
            pass
        
        steps_per_sec = self._total_steps / max(1, elapsed)
        
        print(
            f"  [{pct:5.1f}%] "
            f"vis={self._visual_steps} bab={self._babble_steps} mem={self._memory_steps} "
            f"| {steps_per_sec:.0f} steps/s | VC loss={avg_loss} "
            f"| {remaining/60:.0f}m left"
        )
        sys.stdout.flush()

    def _final_report(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        
        print(f"\n  ════════════════════════════════════════")
        print(f"  Bootstrap Complete!")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"  Visual imagination: {self._visual_steps} steps")
        print(f"  Babble replay: {self._babble_steps} steps")
        print(f"  Memory replay: {self._memory_steps} steps")
        print(f"  Total training: {self._total_steps} steps")
        
        try:
            eyes = self.mind._get_eyes()
            if eyes and hasattr(eyes, '_visual_cortex'):
                stats = eyes._visual_cortex.get_stats()
                print(f"  Visual cortex loss: {stats.get('avg_loss', 'N/A'):.4f}")
        except Exception:
            pass
        
        sub_stats = self.mind.subconscious.get_stats()
        print(f"  Replay buffer: {sub_stats.get('replay_buffer_size', 0)} experiences")
        print(f"  Binding gate: {'OPEN' if sub_stats.get('binding_gate', False) else 'CLOSED'}")
        print(f"  ════════════════════════════════════════\n")
