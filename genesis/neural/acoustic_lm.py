"""
Genesis Mind — Acoustic Language Model

This is the "brain" of the acoustic system. Instead of operating on
text tokens (like GPT), this transformer operates on discrete
acoustic tokens from the VQ codebook.

The principle is identical to how a human infant's brain processes
language: it hears a sequence of sounds and learns to predict what
comes next. Over time, these predictions become so accurate that
the brain can *generate* new sequences — producing speech.

Architecture:
    - Small GPT-style decoder-only transformer
    - 4 layers, 4 attention heads, 128-dim embeddings
    - Vocabulary = 256 (VQ codebook) + 2 special tokens
    - Context window: 256 tokens (~3s of audio at 100 tok/s)
    - Total params: ~500K (trainable on CPU in minutes)

Training:
    - Autoregressive next-token prediction (same as GPT)
    - But on AUDIO tokens, not text tokens
    - Trained in real-time as Genesis hears speech
    - No pre-training — all learning from live interaction
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("genesis.neural.acoustic_lm")

# Special tokens
BOS_TOKEN = 256  # Beginning of sequence
EOS_TOKEN = 257  # End of sequence
PAD_TOKEN = 258  # Padding
VOCAB_SIZE = 259  # 256 codebook + 3 special


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    
    Each position can only attend to itself and previous positions
    (causal mask prevents looking into the future).
    """

    def __init__(self, n_embd: int, n_head: int, max_seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
            .view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class TransformerBlock(nn.Module):
    """Single transformer block: attention + FFN with pre-norm."""

    def __init__(self, n_embd: int, n_head: int, max_seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class AcousticLanguageModel(nn.Module):
    """
    GPT-style transformer language model operating on acoustic tokens.
    
    This is the "Broca's Area" of Genesis — it learns the statistical
    patterns of speech at the acoustic level:
    
        - Which phoneme sequences form valid syllables
        - Which syllable patterns form valid words
        - Which word sequences form valid utterances
        - Prosody and rhythm patterns
    
    All learned from hearing, not from text.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, n_embd: int = 128,
                 n_head: int = 4, n_layer: int = 4,
                 max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_embd = n_embd

        # Token + position embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)
        self.drop = nn.Dropout(dropout)

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, max_seq_len, dropout)
            for _ in range(n_layer)
        ])

        # Final layer norm + output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: output projection shares weights with token embedding
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute logits for next-token prediction.
        
        Args:
            token_ids: (batch, seq_len) integer tensor of token IDs
            
        Returns:
            logits: (batch, seq_len, vocab_size) prediction scores
        """
        B, T = token_ids.shape
        assert T <= self.max_seq_len, f"Sequence too long: {T} > {self.max_seq_len}"

        # Token + position embeddings
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.tok_emb(token_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)

        return logits

    def get_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for a sequence (teacher forcing).
        
        Input tokens: [BOS, t1, t2, ..., tn]
        Target:       [t1,  t2, t3, ..., EOS]
        """
        logits = self.forward(token_ids[:, :-1])  # (B, T-1, V)
        targets = token_ids[:, 1:]  # (B, T-1)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        """
        Autoregressive generation of acoustic tokens.
        
        This is Genesis "speaking" — generating new sound sequences
        based on what it has learned from hearing.
        
        Args:
            prompt: (1, seq_len) starting tokens (e.g., [BOS])
            max_new_tokens: how many tokens to generate
            temperature: randomness (higher = more creative)
            top_k: sample from top-k predictions
            
        Returns:
            generated: (1, seq_len + new_tokens) full token sequence
        """
        self.eval()
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Crop to max context window
            context = generated[:, -self.max_seq_len:]

            # Get predictions
            logits = self.forward(context)
            logits = logits[:, -1, :] / temperature  # Last position

            # Top-k sampling
            if top_k > 0:
                top_values, _ = logits.topk(top_k)
                threshold = top_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Stop at EOS
            if next_token.item() == EOS_TOKEN:
                break

            generated = torch.cat([generated, next_token], dim=1)

        return generated


class AcousticBrain:
    """
    High-level wrapper around the AcousticLanguageModel.
    Handles training, generation, and statistics.
    """

    def __init__(self, n_embd: int = 128, n_head: int = 4,
                 n_layer: int = 4, max_seq_len: int = 256,
                 lr: float = 0.0003):
        self.model = AcousticLanguageModel(
            n_embd=n_embd, n_head=n_head, n_layer=n_layer,
            max_seq_len=max_seq_len,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )
        self._total_sequences_heard = 0
        self._total_tokens_seen = 0
        self._running_loss = 0.0
        self._loss_count = 0

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Acoustic Language Model initialized (%d params, %d layers, %d heads)",
            total_params, n_layer, n_head,
        )

    def learn_from_tokens(self, token_sequence: List[int]) -> float:
        """
        Train on a single heard token sequence.
        
        Called whenever Genesis hears speech — the tokens from the
        VQ codebook are fed here as a training sequence.
        """
        # Wrap with BOS and EOS
        tokens = [BOS_TOKEN] + token_sequence + [EOS_TOKEN]
        
        # Truncate if too long
        if len(tokens) > self.model.max_seq_len:
            tokens = tokens[:self.model.max_seq_len]

        tensor = torch.tensor([tokens], dtype=torch.long)

        self.model.train()
        loss = self.model.get_loss(tensor)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self._total_sequences_heard += 1
        self._total_tokens_seen += len(tokens)
        self._running_loss += loss_val
        self._loss_count += 1

        return loss_val

    def generate_response(self, context_tokens: Optional[List[int]] = None,
                          max_tokens: int = 50,
                          temperature: float = 0.8) -> List[int]:
        """
        Generate a response sequence of acoustic tokens.
        
        This is Genesis "thinking" what to say — producing a new
        sequence of audio tokens that can be decoded into sound.
        """
        if context_tokens:
            prompt = torch.tensor([[BOS_TOKEN] + context_tokens[-100:]], dtype=torch.long)
        else:
            prompt = torch.tensor([[BOS_TOKEN]], dtype=torch.long)

        generated = self.model.generate(
            prompt, max_new_tokens=max_tokens 
            , temperature=temperature,
        )

        # Strip BOS and any special tokens
        result = [t for t in generated[0].tolist() if t < 256]
        return result

    def get_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_stats(self) -> Dict:
        avg_loss = self._running_loss / max(1, self._loss_count)
        return {
            "params": self.get_params(),
            "total_sequences_heard": self._total_sequences_heard,
            "total_tokens_seen": self._total_tokens_seen,
            "avg_loss": round(avg_loss, 4),
            "vocab_size": VOCAB_SIZE,
        }

    def save_weights(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stats': {
                'total_sequences_heard': self._total_sequences_heard,
                'total_tokens_seen': self._total_tokens_seen,
                'running_loss': self._running_loss,
                'loss_count': self._loss_count,
            },
        }, path)

    def load_weights(self, path):
        try:
            ckpt = torch.load(path, weights_only=False)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            stats = ckpt.get('stats', {})
            self._total_sequences_heard = stats.get('total_sequences_heard', 0)
            self._total_tokens_seen = stats.get('total_tokens_seen', 0)
            self._running_loss = stats.get('running_loss', 0.0)
            self._loss_count = stats.get('loss_count', 0)
            logger.info("Loaded Acoustic LM weights (heard %d sequences)",
                        self._total_sequences_heard)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not load Acoustic LM weights: %s", e)


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Acoustic Language Model Test")
    print("=" * 60)

    brain = AcousticBrain(n_embd=128, n_head=4, n_layer=4)
    print(f"\nParams: {brain.get_params():,}")

    # Simulate heard token sequences (from VQ codebook)
    print("\n--- Training on simulated audio tokens ---")
    import random
    for i in range(20):
        # Simulate a "sentence" of 20-50 audio tokens
        seq = [random.randint(0, 255) for _ in range(random.randint(20, 50))]
        loss = brain.learn_from_tokens(seq)
        if i % 5 == 0:
            print(f"  Step {i+1}: loss={loss:.4f}")

    # Generate a response
    print("\n--- Generation Test ---")
    response = brain.generate_response(max_tokens=30, temperature=0.8)
    print(f"  Generated {len(response)} tokens: {response[:20]}...")

    # Generate with context
    context = [42, 100, 200, 50, 75]
    response2 = brain.generate_response(context_tokens=context, max_tokens=20)
    print(f"  With context: {len(response2)} tokens: {response2[:20]}...")

    print(f"\n  Stats: {brain.get_stats()}")
    print("Acoustic Language Model test PASSED")
