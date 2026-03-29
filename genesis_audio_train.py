"""
Genesis Audio Training — End-to-End VQ-VAE with fresh encoder

Fixes codebook collapse by creating an entirely new encoder + codebook
from scratch and training them jointly on real speech data.
"""

import logging
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
logger = logging.getLogger(__name__)

GENESIS_HOME = Path.home() / ".genesis"
AUDIO_WEIGHTS = GENESIS_HOME / "acoustic_weights"


def load_audio_files(max_clips: int = 5000) -> list:
    """Load LJSpeech wav files as 16kHz numpy arrays."""
    wav_dir = GENESIS_HOME / "datasets" / "LJSpeech-1.1" / "wavs"
    if not wav_dir.exists():
        logger.error("LJSpeech not found at %s", wav_dir)
        sys.exit(1)

    try:
        import soundfile as sf
    except ImportError:
        import os; os.system(f"{sys.executable} -m pip install soundfile -q")
        import soundfile as sf

    wav_files = sorted(wav_dir.glob("*.wav"))[:max_clips]
    logger.info("Loading %d audio files...", len(wav_files))

    audios = []
    for wf in wav_files:
        try:
            audio, sr = sf.read(str(wf))
            audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                ratio = 16000 / sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_len),
                    np.arange(len(audio)), audio,
                ).astype(np.float32)
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.7
            audios.append(audio)
        except Exception:
            pass

    logger.info("Loaded %d clips", len(audios))
    return audios


def restart_dead_codes(vq, encoder_outputs):
    """Replace dead codebook entries with random encoder outputs."""
    with torch.no_grad():
        usage = vq._codebook_usage
        dead = (usage == 0)
        n_dead = dead.sum().item()
        if n_dead == 0 or len(encoder_outputs) == 0:
            return 0

        n = min(n_dead, len(encoder_outputs))
        idx = torch.randperm(len(encoder_outputs))[:n]
        replacements = encoder_outputs[idx] + torch.randn(n, vq.latent_dim) * 0.02

        dead_idx = torch.where(dead)[0][:n]
        vq.embedding.weight.data[dead_idx] = replacements
        vq.ema_weight[dead_idx] = replacements
        vq.ema_count[dead_idx] = 1.0
        return n


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print(  "║   GENESIS AUDIO TRAINING — Fresh VQ-VAE Training      ║")
    print(  "╚══════════════════════════════════════════════════════════╝\n")

    audios = load_audio_files(args.clips)
    if not audios:
        return

    # Import genesis modules
    sys.path.insert(0, str(Path(__file__).parent))
    from genesis.neural.auditory_cortex import AuditoryCortex, A1Encoder, MelFilterBank
    from genesis.neural.vq_codebook import VQCodebook
    from genesis.neural.acoustic_lm import AcousticBrain

    # Create FRESH components — no loading stale weights
    logger.info("Creating fresh encoder + codebook from scratch...")
    mel_filter = MelFilterBank(sample_rate=16000, n_mels=80)
    encoder = A1Encoder(n_mels=80, latent_dim=64)
    vq = VQCodebook(codebook_size=256, latent_dim=64, commitment_cost=0.25, ema_decay=0.99)

    # Load existing Acoustic LM weights (keep learned speech patterns)
    acoustic_brain = AcousticBrain(n_embd=128, n_head=4, n_layer=4)
    lm_path = AUDIO_WEIGHTS / "acoustic_lm.pt"
    if lm_path.exists():
        acoustic_brain.load_weights(lm_path)
        logger.info("Loaded existing Acoustic LM (preserving learned patterns)")

    # Optimizer for encoder only — VQ uses EMA
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(audios), eta_min=1e-5
    )

    # Warm start: initialize codebook from first batch of encoder outputs
    logger.info("Warm-starting codebook from first 200 audio clips...")
    encoder.eval()
    warmup_outputs = []
    with torch.no_grad():
        for audio in audios[:200]:
            wav_t = torch.tensor(audio, dtype=torch.float32)
            mel = mel_filter(wav_t)
            latent = encoder(mel.unsqueeze(0))  # (1, 64, T)
            z_flat = latent.permute(0, 2, 1).reshape(-1, 64)  # (T, 64)
            warmup_outputs.append(z_flat)

    all_outputs = torch.cat(warmup_outputs, dim=0)
    logger.info("  Encoder outputs shape: %s, mean=%.4f, std=%.4f",
                all_outputs.shape, all_outputs.mean().item(), all_outputs.std().item())

    # Initialize codebook entries from kmeans-like sampling of actual outputs
    with torch.no_grad():
        # Sample 256 random encoder outputs as initial codebook entries
        indices = torch.randperm(len(all_outputs))[:256]
        init_entries = all_outputs[indices] + torch.randn(256, 64) * 0.01
        vq.embedding.weight.copy_(init_entries)
        vq.ema_weight.copy_(init_entries)
        vq.ema_count.fill_(1.0)

    # Verify: how many unique tokens now?
    with torch.no_grad():
        test_wav = torch.tensor(audios[0], dtype=torch.float32)
        test_mel = mel_filter(test_wav)
        test_latent = encoder(test_mel.unsqueeze(0))
        _, test_ids, _ = vq(test_latent)
        unique_after_init = len(set(test_ids[0].tolist()))
        logger.info("  After warm-start init: %d unique tokens (was 1)", unique_after_init)

    # ═══ TRAINING LOOP ═══
    print(f"\n=== TRAINING ({args.epochs} epochs × {len(audios)} clips) ===\n")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        vq.train()

        # Reset usage counter per epoch
        vq._codebook_usage.zero_()
        vq._total_quantizations = 0

        total_vq_loss = 0.0
        indices = np.random.permutation(len(audios))
        epoch_outputs = []

        for step, i in enumerate(indices):
            audio = audios[i]
            wav_t = torch.tensor(audio, dtype=torch.float32)
            mel = mel_filter(wav_t)
            latent = encoder(mel.unsqueeze(0))  # (1, 64, T)

            z_q, token_ids, vq_loss = vq(latent)

            # Check for NaN and skip
            if torch.isnan(vq_loss):
                continue

            optimizer.zero_grad()
            vq_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_vq_loss += vq_loss.item()

            # Collect outputs for dead code restart
            if step < 200:
                z_flat = latent.permute(0, 2, 1).reshape(-1, 64).detach()
                epoch_outputs.append(z_flat)

            # Train LM too
            tokens = token_ids[0].detach().tolist()
            acoustic_brain.learn_from_tokens(tokens)

            if (step + 1) % 500 == 0:
                util = vq.get_codebook_utilization() * 100
                active = (vq._codebook_usage > 0).sum().item()
                unique = len(set(tokens))
                cur_lr = scheduler.get_last_lr()[0]
                logger.info(
                    "  Epoch %d [%d/%d] | VQ loss=%.4f | util=%.1f%% (%d active) | "
                    "%d unique | lr=%.6f",
                    epoch, step + 1, len(audios),
                    total_vq_loss / (step + 1),
                    util, active, unique, cur_lr,
                )

        # Dead code restart
        if epoch_outputs:
            all_o = torch.cat(epoch_outputs, dim=0)
            n_restart = restart_dead_codes(vq, all_o)
            if n_restart > 0:
                logger.info("  ↻ Restarted %d dead codes from encoder outputs", n_restart)

        util = vq.get_codebook_utilization() * 100
        active = (vq._codebook_usage > 0).sum().item()
        logger.info(
            "  Epoch %d DONE | avg loss=%.4f | util=%.1f%% (%d/256 active)",
            epoch, total_vq_loss / max(1, len(audios)), util, active,
        )

        # Save checkpoint
        AUDIO_WEIGHTS.mkdir(parents=True, exist_ok=True)

        # Save encoder as auditory cortex weights
        torch.save(encoder.state_dict(), AUDIO_WEIGHTS / "auditory_cortex.pt")

        # Save VQ codebook
        torch.save({
            'embedding': vq.embedding.state_dict(),
            'ema_count': vq.ema_count,
            'ema_weight': vq.ema_weight,
            'usage': vq._codebook_usage,
            'total_quantizations': vq._total_quantizations,
        }, AUDIO_WEIGHTS / "vq_codebook.pt")

        # Save acoustic LM
        acoustic_brain.save_weights(AUDIO_WEIGHTS / "acoustic_lm.pt")

        logger.info("  ✓ Checkpoint saved (epoch %d)\n", epoch)

    elapsed = time.time() - start

    # Final test
    encoder.eval()
    with torch.no_grad():
        test_latent = encoder(test_mel.unsqueeze(0))
        _, test_ids, _ = vq(test_latent)
        tokens = test_ids[0].tolist()
        unique = len(set(tokens))

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  VQ utilization: {vq.get_codebook_utilization()*100:.1f}%")
    print(f"  Active codes: {(vq._codebook_usage > 0).sum().item()}/256")
    print(f"  Test clip: {len(tokens)} tokens, {unique} unique")
    print(f"  Token sample: {tokens[:30]}")
    print(f"  Acoustic LM: {acoustic_brain.get_stats()['total_sequences_heard']:,} seqs")
    print(f"  Training time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Weights: {AUDIO_WEIGHTS}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
