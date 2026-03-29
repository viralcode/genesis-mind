"""
Genesis Voice Rehabilitation — Fix the Neural Vocoder

The neural vocoder produces noise/vibration instead of speech because:
  1. CODEBOOK COLLAPSE: Only 51/256 VQ entries are alive (80% dead)
  2. MEL RECONSTRUCTOR: Barely trained — can't map VQ→Mel accurately
  3. ACOUSTIC LM: Generates random token sequences, not speech patterns

This script fixes all three by running a focused rehabilitation loop:

  Phase 1: CODEBOOK REVIVAL — Restart dead entries and spread encoder
  Phase 2: RECONSTRUCTION — Train VQ→Mel→Waveform with real speech
  Phase 3: ACOUSTIC LM — Train the language model on real token sequences

Usage:
    python genesis_voice_rehab.py                  # Full rehab (all phases)
    python genesis_voice_rehab.py --phase 1        # Just codebook fix
    python genesis_voice_rehab.py --phase 2        # Just reconstruction
    python genesis_voice_rehab.py --phase 3        # Just acoustic LM
    python genesis_voice_rehab.py --epochs 10      # More epochs
    python genesis_voice_rehab.py --test            # Test output quality
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("genesis.voice_rehab")

GENESIS_HOME = Path.home() / ".genesis"
AUDIO_WEIGHTS = GENESIS_HOME / "acoustic_weights"
DATASETS = GENESIS_HOME / "datasets"
LIBRI_DIR = DATASETS / "LibriSpeech"


# =============================================================================
# Audio Loading
# =============================================================================

def load_flac(path: Path) -> np.ndarray:
    """Load a FLAC file and return 16kHz mono float32 numpy array."""
    try:
        import scipy.io.wavfile as wavfile
        import subprocess
        import tempfile

        # Convert FLAC to WAV using ffmpeg/afconvert
        tmp_wav = Path(tempfile.gettempdir()) / f"genesis_rehab_{os.getpid()}.wav"
        
        # Try ffmpeg first, then afconvert (macOS built-in)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(tmp_wav)],
                capture_output=True, timeout=10,
            )
        except FileNotFoundError:
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                 str(path), str(tmp_wav)],
                capture_output=True, timeout=10,
            )

        sr, data = wavfile.read(str(tmp_wav))
        tmp_wav.unlink(missing_ok=True)

        # Convert to float32
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        
        # Mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        # Normalize
        peak = np.abs(data).max()
        if peak > 1e-6:
            data = data / peak * 0.95

        return data

    except Exception as e:
        logger.warning("Failed to load %s: %s", path.name, e)
        return None


def load_librispeech(max_files: int = 2000) -> list:
    """Load LibriSpeech FLAC files as numpy arrays."""
    flac_files = sorted(LIBRI_DIR.rglob("*.flac"))
    if not flac_files:
        logger.error("No LibriSpeech data found at %s", LIBRI_DIR)
        sys.exit(1)

    logger.info("Found %d FLAC files, loading up to %d...", len(flac_files), max_files)

    # Shuffle and limit
    rng = np.random.default_rng(42)
    rng.shuffle(flac_files)
    flac_files = flac_files[:max_files]

    audios = []
    for i, f in enumerate(flac_files):
        audio = load_flac(f)
        if audio is not None and len(audio) >= 3200:  # At least 0.2s
            # Trim to max 10s to save memory
            if len(audio) > 160000:
                audio = audio[:160000]
            audios.append(audio)

        if (i + 1) % 200 == 0:
            logger.info("  Loaded %d/%d files (%d usable)", i + 1, len(flac_files), len(audios))

    logger.info("✓ Loaded %d audio clips from LibriSpeech", len(audios))
    return audios


# =============================================================================
# Load Components
# =============================================================================

def load_components():
    """Load all acoustic pipeline components with their trained weights."""
    from genesis.neural.auditory_cortex import A1Encoder, MelFilterBank
    from genesis.neural.vq_codebook import VQCodebook
    from genesis.neural.acoustic_lm import AcousticBrain
    from genesis.neural.neural_vocoder import NeuralVocoder

    mel_filter = MelFilterBank(sample_rate=16000, n_mels=80)
    encoder = A1Encoder(n_mels=80, latent_dim=64)
    vq = VQCodebook(codebook_size=256, latent_dim=64, commitment_cost=0.25, ema_decay=0.99)
    acoustic_brain = AcousticBrain(n_embd=128, n_head=4, n_layer=4)
    vocoder = NeuralVocoder(latent_dim=64, n_mels=80, lr=0.0005)

    # Load existing weights
    enc_path = AUDIO_WEIGHTS / "auditory_cortex.pt"
    vq_path = AUDIO_WEIGHTS / "vq_codebook.pt"
    lm_path = AUDIO_WEIGHTS / "acoustic_lm.pt"
    voc_path = AUDIO_WEIGHTS / "neural_vocoder.pt"

    if enc_path.exists():
        encoder.load_state_dict(torch.load(enc_path, map_location='cpu', weights_only=True))
        logger.info("  ✓ Encoder loaded")
    if vq_path.exists():
        ckpt = torch.load(vq_path, map_location='cpu', weights_only=False)
        vq.embedding.load_state_dict(ckpt['embedding'])
        vq.ema_count = ckpt['ema_count']
        vq.ema_weight = ckpt['ema_weight']
        vq._codebook_usage = ckpt.get('usage', vq._codebook_usage)
        active = (vq._codebook_usage > 0).sum().item()
        logger.info("  ✓ VQ Codebook loaded (%d/256 alive)", active)
    if lm_path.exists():
        acoustic_brain.load_weights(lm_path)
        logger.info("  ✓ Acoustic LM loaded")
    if voc_path.exists():
        vocoder.load_weights(voc_path)
        logger.info("  ✓ Vocoder loaded")

    return mel_filter, encoder, vq, acoustic_brain, vocoder


def save_checkpoint(encoder, vq, acoustic_brain, vocoder, tag=""):
    """Save all weights."""
    AUDIO_WEIGHTS.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), AUDIO_WEIGHTS / "auditory_cortex.pt")
    torch.save({
        'embedding': vq.embedding.state_dict(),
        'ema_count': vq.ema_count,
        'ema_weight': vq.ema_weight,
        'usage': vq._codebook_usage,
        'total_quantizations': vq._total_quantizations,
    }, AUDIO_WEIGHTS / "vq_codebook.pt")
    acoustic_brain.save_weights(AUDIO_WEIGHTS / "acoustic_lm.pt")
    vocoder.save_weights(AUDIO_WEIGHTS / "neural_vocoder.pt")
    logger.info("  ✓ Checkpoint saved %s", tag)


# =============================================================================
# PHASE 1: Codebook Revival
# =============================================================================

def phase1_codebook_revival(encoder, vq, mel_filter, audios, epochs=5):
    """
    Fix the codebook collapse problem.
    
    Strategy:
    1. Reset ALL dead codebook entries to random encoder outputs + noise
    2. Lower EMA decay temporarily so dead entries get a chance to capture inputs
    3. Use diversity regularization loss to spread encoder outputs
    4. Aggressive dead code restart every 200 steps
    """
    logger.info("\n" + "═" * 60)
    logger.info("  PHASE 1: CODEBOOK REVIVAL")
    logger.info("  Target: 200+ / 256 alive codes (currently %d)",
                (vq._codebook_usage > 0).sum().item())
    logger.info("═" * 60 + "\n")

    # Reset usage tracking
    vq._codebook_usage.zero_()

    # Use a faster EMA decay during revival to let new entries catch up
    original_decay = vq.ema_decay
    vq.ema_decay = 0.95  # Faster adaptation

    # Stronger commitment cost forces encoder to spread outputs
    original_commitment = vq.commitment_cost
    vq.commitment_cost = 0.5

    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(audios), eta_min=1e-5,
    )

    for epoch in range(1, epochs + 1):
        encoder.train()
        vq.train()
        vq._codebook_usage.zero_()

        total_loss = 0.0
        epoch_latents = []
        indices = np.random.permutation(len(audios))

        for step, i in enumerate(indices):
            audio = audios[i]
            wav_t = torch.tensor(audio, dtype=torch.float32)
            mel = mel_filter(wav_t)
            latent = encoder(mel.unsqueeze(0))  # (1, 64, T)

            z_q, token_ids, vq_loss = vq(latent)

            if torch.isnan(vq_loss):
                continue

            # Add diversity regularization: encourage encoder outputs to be spread
            # across the embedding space by penalizing low-variance outputs
            z_flat = latent.permute(0, 2, 1).reshape(-1, 64)
            diversity_loss = -z_flat.std(dim=0).mean() * 0.1  # Encourage variance

            loss = vq_loss + diversity_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += vq_loss.item()

            # Collect latents for dead restart
            if step < 500:
                epoch_latents.append(z_flat.detach())

            # Aggressive dead code restart every 200 steps
            if (step + 1) % 200 == 0 and epoch_latents:
                all_latents = torch.cat(epoch_latents, dim=0)
                with torch.no_grad():
                    dead = (vq._codebook_usage == 0)
                    n_dead = dead.sum().item()
                    if n_dead > 0:
                        n = min(n_dead, len(all_latents))
                        # Use k-means-like diverse initialization
                        idx = torch.randperm(len(all_latents))[:n]
                        replacements = all_latents[idx] + torch.randn(n, 64) * 0.05
                        dead_idx = torch.where(dead)[0][:n]
                        vq.embedding.weight.data[dead_idx] = replacements
                        vq.ema_weight[dead_idx] = replacements
                        vq.ema_count[dead_idx] = 1.0
                        vq._codebook_usage[dead_idx] = 1.0
                        logger.info("    ↻ Step %d: Restarted %d dead codes (%d/%d now alive)",
                                    step + 1, n, 256 - n_dead + n, 256)

            if (step + 1) % 500 == 0:
                active = (vq._codebook_usage > 0).sum().item()
                logger.info("  Epoch %d [%d/%d] | VQ=%.4f | alive=%d/256 (%.0f%%)",
                            epoch, step + 1, len(audios), total_loss / (step + 1),
                            active, active / 256 * 100)

        # End-of-epoch restart
        if epoch_latents:
            all_latents = torch.cat(epoch_latents, dim=0)
            with torch.no_grad():
                dead = (vq._codebook_usage == 0)
                n_dead = dead.sum().item()
                if n_dead > 0:
                    n = min(n_dead, len(all_latents))
                    idx = torch.randperm(len(all_latents))[:n]
                    replacements = all_latents[idx] + torch.randn(n, 64) * 0.03
                    dead_idx = torch.where(dead)[0][:n]
                    vq.embedding.weight.data[dead_idx] = replacements
                    vq.ema_weight[dead_idx] = replacements
                    vq.ema_count[dead_idx] = 1.0
                    logger.info("  ↻ End-of-epoch: Restarted %d dead codes", n)

        active = (vq._codebook_usage > 0).sum().item()
        logger.info("  ✓ Epoch %d DONE | avg VQ=%.4f | alive=%d/256 (%.0f%%)\n",
                     epoch, total_loss / max(1, len(audios)), active, active / 256 * 100)

    # Restore normal parameters
    vq.ema_decay = original_decay
    vq.commitment_cost = original_commitment

    active = (vq._codebook_usage > 0).sum().item()
    logger.info("  PHASE 1 COMPLETE: %d/256 codebook entries alive (%.0f%%)",
                active, active / 256 * 100)
    return active


# =============================================================================
# PHASE 2: Mel Reconstruction Training
# =============================================================================

def phase2_reconstruction(encoder, vq, vocoder, mel_filter, audios, epochs=5):
    """
    Train the vocoder's mel reconstructor with supervised reconstruction loss.
    
    Pipeline: Audio → Mel → Encoder → VQ → MelReconstructor → Predicted Mel
    Loss: MSE(Predicted Mel, Original Mel)
    
    This teaches the vocoder to faithfully reconstruct speech from VQ codes.
    """
    logger.info("\n" + "═" * 60)
    logger.info("  PHASE 2: MEL RECONSTRUCTION TRAINING")
    logger.info("  Teaching vocoder to reconstruct speech from VQ codes...")
    logger.info("═" * 60 + "\n")

    # Use a fresh optimizer for the vocoder with a good learning rate
    voc_optimizer = torch.optim.Adam(
        vocoder.mel_reconstructor.parameters(), lr=0.001, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        voc_optimizer, T_max=epochs * len(audios), eta_min=1e-5,
    )

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        vocoder.mel_reconstructor.train()
        encoder.eval()
        vq.eval()

        total_loss = 0.0
        n_valid = 0
        indices = np.random.permutation(len(audios))

        for step, i in enumerate(indices):
            audio = audios[i]
            wav_t = torch.tensor(audio, dtype=torch.float32)
            mel_target = mel_filter(wav_t)  # (80, T)

            with torch.no_grad():
                latent = encoder(mel_target.unsqueeze(0))  # (1, 64, T')
                z_q, token_ids, _ = vq(latent)  # (1, 64, T')

            # Reconstruct mel from VQ codes
            mel_pred = vocoder.mel_reconstructor(z_q)  # (1, 80, T'*4)

            # Align sizes (the mel reconstructor upsamples by 4x)
            T_pred = mel_pred.shape[-1]
            T_target = mel_target.shape[-1]

            if T_pred >= T_target:
                mel_pred_aligned = mel_pred[:, :, :T_target]
                mel_target_aligned = mel_target.unsqueeze(0)
            else:
                mel_pred_aligned = mel_pred
                mel_target_aligned = mel_target.unsqueeze(0)[:, :, :T_pred]

            # MSE reconstruction loss
            recon_loss = F.mse_loss(mel_pred_aligned, mel_target_aligned)

            # Spectral convergence loss (encourages sharper spectral features)
            spectral_loss = torch.norm(mel_target_aligned - mel_pred_aligned) / (
                torch.norm(mel_target_aligned) + 1e-8
            )

            loss = recon_loss + 0.5 * spectral_loss

            if torch.isnan(loss):
                continue

            voc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vocoder.mel_reconstructor.parameters(), 1.0)
            voc_optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_valid += 1

            if (step + 1) % 500 == 0:
                avg = total_loss / n_valid
                logger.info("  Epoch %d [%d/%d] | recon=%.4f | lr=%.6f",
                            epoch, step + 1, len(audios), avg,
                            scheduler.get_last_lr()[0])

        avg_loss = total_loss / max(1, n_valid)
        if avg_loss < best_loss:
            best_loss = avg_loss
            tag = " ★ BEST"
        else:
            tag = ""

        logger.info("  ✓ Epoch %d DONE | avg recon=%.4f%s\n", epoch, avg_loss, tag)

    logger.info("  PHASE 2 COMPLETE: best reconstruction loss = %.4f", best_loss)


# =============================================================================
# PHASE 3: Acoustic LM Training on Real Speech
# =============================================================================

def phase3_acoustic_lm(encoder, vq, acoustic_brain, mel_filter, audios, epochs=3):
    """
    Train the Acoustic Language Model on real speech token sequences.
    
    This teaches the LM to generate token sequences that follow
    natural speech patterns instead of random noise.
    """
    logger.info("\n" + "═" * 60)
    logger.info("  PHASE 3: ACOUSTIC LM ON REAL SPEECH")
    logger.info("  Teaching the language model what real speech sounds like...")
    logger.info("═" * 60 + "\n")

    encoder.eval()
    vq.eval()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_valid = 0
        indices = np.random.permutation(len(audios))

        for step, i in enumerate(indices):
            audio = audios[i]
            wav_t = torch.tensor(audio, dtype=torch.float32)
            mel = mel_filter(wav_t)

            with torch.no_grad():
                latent = encoder(mel.unsqueeze(0))
                z_q, token_ids, _ = vq(latent)

            tokens = token_ids[0].tolist()

            # Train the acoustic brain to predict these real speech tokens
            loss = acoustic_brain.learn_from_tokens(tokens)
            total_loss += loss
            n_valid += 1

            if (step + 1) % 500 == 0:
                avg = total_loss / n_valid
                unique_tokens = len(set(tokens))
                logger.info("  Epoch %d [%d/%d] | LM loss=%.4f | unique tokens=%d",
                            epoch, step + 1, len(audios), avg, unique_tokens)

        avg_loss = total_loss / max(1, n_valid)
        stats = acoustic_brain.get_stats()
        logger.info("  ✓ Epoch %d DONE | avg LM=%.4f | heard %d seqs, %d tokens\n",
                     epoch, avg_loss, stats['total_sequences_heard'],
                     stats['total_tokens_seen'])

    logger.info("  PHASE 3 COMPLETE")


# =============================================================================
# Test: Generate and play speech
# =============================================================================

def test_output(encoder, vq, acoustic_brain, vocoder, mel_filter, audios):
    """
    Test the voice pipeline by:
    1. Reconstructing a real LibriSpeech clip (should sound like speech)
    2. Generating spontaneous tokens and playing them
    3. Comparing both outputs
    """
    logger.info("\n" + "═" * 60)
    logger.info("  VOICE QUALITY TEST")
    logger.info("═" * 60 + "\n")

    import scipy.io.wavfile as wavfile
    import subprocess
    output_dir = GENESIS_HOME / "voice_test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- TEST 1: Reconstruction ---
    logger.info("  TEST 1: Reconstructing a real LibriSpeech clip...")
    test_audio = audios[0]

    # Save original
    orig_path = output_dir / "original.wav"
    wavfile.write(str(orig_path), 16000, (test_audio * 32767).astype(np.int16))
    logger.info("    Original saved: %s (%.1fs)", orig_path, len(test_audio) / 16000)

    # Encode → VQ → Reconstruct
    encoder.eval()
    vq.eval()
    vocoder.mel_reconstructor.eval()

    wav_t = torch.tensor(test_audio, dtype=torch.float32)
    mel = mel_filter(wav_t)
    with torch.no_grad():
        latent = encoder(mel.unsqueeze(0))
        z_q, token_ids, _ = vq(latent)

    active = len(set(token_ids[0].tolist()))
    logger.info("    Encoded → %d tokens (%d unique)", len(token_ids[0]), active)

    # Reconstruct waveform
    waveform = vocoder.synthesize_from_embeddings(z_q)
    recon_path = output_dir / "reconstructed.wav"
    waveform_int = (waveform * 32767).astype(np.int16)
    wavfile.write(str(recon_path), 16000, waveform_int)
    logger.info("    Reconstructed saved: %s (%.1fs)", recon_path, len(waveform) / 16000)

    # --- TEST 2: Spontaneous generation ---
    logger.info("\n  TEST 2: Generating spontaneous speech tokens...")
    context_tokens = token_ids[0][:50].tolist()  # Use real context
    generated = acoustic_brain.generate_response(
        context_tokens=context_tokens, max_tokens=60, temperature=0.7,
    )
    logger.info("    Generated %d tokens: %s...", len(generated), generated[:15])

    gen_ids = torch.tensor([generated], dtype=torch.long)
    gen_embeddings = vq.tokens_to_embeddings(gen_ids)
    gen_waveform = vocoder.synthesize_from_embeddings(gen_embeddings)

    gen_path = output_dir / "generated.wav"
    gen_int = (gen_waveform * 32767).astype(np.int16)
    wavfile.write(str(gen_path), 16000, gen_int)
    logger.info("    Generated saved: %s (%.1fs)", gen_path, len(gen_waveform) / 16000)

    # --- Playback ---
    logger.info("\n  🔊 Playing original...")
    subprocess.run(["afplay", str(orig_path)], timeout=15)
    time.sleep(0.5)

    logger.info("  🔊 Playing reconstructed...")
    subprocess.run(["afplay", str(recon_path)], timeout=15)
    time.sleep(0.5)

    logger.info("  🔊 Playing generated...")
    subprocess.run(["afplay", str(gen_path)], timeout=15)

    # Stats
    active = (vq._codebook_usage > 0).sum().item()
    logger.info("\n  ═══ RESULTS ═══")
    logger.info("  Codebook utilization: %d/256 (%.0f%%)", active, active / 256 * 100)
    logger.info("  Output files in: %s", output_dir)
    logger.info("  Original:       %s", orig_path)
    logger.info("  Reconstructed:  %s", recon_path)
    logger.info("  Generated:      %s", gen_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Genesis Voice Rehabilitation")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1=codebook, 2=reconstruction, 3=acoustic LM, 0=all)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per phase (default: 5)")
    parser.add_argument("--max-files", type=int, default=2000,
                        help="Max LibriSpeech files to load (default: 2000)")
    parser.add_argument("--test", action="store_true",
                        help="Run voice quality test after training")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run the test (no training)")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  GENESIS VOICE REHABILITATION")
    print("  Fixing the neural vocoder — from noise to speech")
    print("═" * 60 + "\n")

    start = time.time()

    # Load data
    audios = load_librispeech(args.max_files)

    # Load components
    logger.info("\nLoading acoustic pipeline components...")
    mel_filter, encoder, vq, acoustic_brain, vocoder = load_components()

    # Show initial state
    active = (vq._codebook_usage > 0).sum().item()
    logger.info("\n  INITIAL STATE:")
    logger.info("    Codebook: %d/256 alive (%.0f%%)", active, active / 256 * 100)
    logger.info("    Audio clips: %d", len(audios))
    logger.info("    Vocoder params: %d", vocoder.get_params())
    lm_stats = acoustic_brain.get_stats()
    logger.info("    Acoustic LM: heard %d seqs, %d tokens\n",
                lm_stats['total_sequences_heard'], lm_stats['total_tokens_seen'])

    if args.test_only:
        test_output(encoder, vq, acoustic_brain, vocoder, mel_filter, audios)
        return

    # Run phases
    if args.phase == 0 or args.phase == 1:
        phase1_codebook_revival(encoder, vq, mel_filter, audios, epochs=args.epochs)
        save_checkpoint(encoder, vq, acoustic_brain, vocoder, tag="(after Phase 1)")

    if args.phase == 0 or args.phase == 2:
        phase2_reconstruction(encoder, vq, vocoder, mel_filter, audios, epochs=args.epochs)
        save_checkpoint(encoder, vq, acoustic_brain, vocoder, tag="(after Phase 2)")

    if args.phase == 0 or args.phase == 3:
        phase3_acoustic_lm(encoder, vq, acoustic_brain, mel_filter, audios, epochs=args.epochs)
        save_checkpoint(encoder, vq, acoustic_brain, vocoder, tag="(after Phase 3)")

    elapsed = time.time() - start
    active = (vq._codebook_usage > 0).sum().item()

    print("\n" + "═" * 60)
    print(f"  VOICE REHABILITATION COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Codebook: {active}/256 alive ({active/256*100:.0f}%)")
    print(f"  Weights saved: {AUDIO_WEIGHTS}")
    print("═" * 60 + "\n")

    # Auto-test if requested
    if args.test:
        test_output(encoder, vq, acoustic_brain, vocoder, mel_filter, audios)


if __name__ == "__main__":
    main()
