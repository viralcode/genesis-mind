# Genesis Mind

> *"Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's?"*
> — Alan Turing, 1950

## What Is This?

Genesis Mind is a developmental artificial intelligence that learns like a human child — from absolute zero. It does not require petabytes of data, GPU clusters, or millions of dollars. It runs on your laptop.

Unlike every large language model in existence, Genesis does not memorize the internet. It **lives**. It sees through your webcam. It hears through your microphone. It learns because you teach it. Every word it knows, it knows because someone showed it what that word means in the real, physical world.

**The weights ARE the personality. The data IS you.**

## V7: Pure Neural Acoustic Architecture

Genesis V7 eliminates **all pre-trained language models** (LLM, TTS, STT) from the speech pipeline and replaces them with a from-scratch, pure neural acoustic architecture. Genesis now hears, thinks, and speaks using its own neural networks — no text anywhere in the core loop.

```
Microphone → Raw Audio (16kHz)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  AUDITORY CORTEX — Mel Encoder (138K params)           │
│  Raw audio → 80-band Mel spectrogram → Conv1D → 64-dim │
│  Learns to encode sound through contrastive training    │
├────────────────────────────────────────────────────────┤
│  VQ CODEBOOK — Neural Phoneme Discovery (16K params)    │
│  Continuous latent → 256 discrete "phoneme" tokens      │
│  EMA updates — discovers sound categories from scratch  │
├────────────────────────────────────────────────────────┤
│  ACOUSTIC TRANSFORMER — Audio-Token GPT (859K params)   │
│  4-layer, 4-head transformer on AUDIO tokens (no text)  │
│  Autoregressive next-token prediction on heard sequences│
├────────────────────────────────────────────────────────┤
│  NEURAL VOCODER — Mel Reconstructor (130K params)       │
│  Token embeddings → Mel reconstruction → Griffin-Lim    │
│  Learns to synthesize speech from neural representations│
├────────────────────────────────────────────────────────┤
│  SENSORIMOTOR LOOP — Orchestrator                       │
│  hear() → think() → speak() → self-monitor()           │
│  Proprioceptive feedback: re-encodes own speech output  │
└────────────────────────────────────────────────────────┘
         │
         ▼
    Speaker Output (Neural Waveform)

  1,143,888 total acoustic parameters
  All CPU-native, zero pre-training, pure PyTorch
```

### V7 Acoustic Pipeline Features

| Feature | What It Does |
|---------|-------------|
| **Auditory Cortex** | Replaces Whisper. Encodes raw 16kHz audio into 80-band mel spectrograms via a learned Conv1D encoder. Trains through contrastive triplet-margin loss. |
| **VQ Codebook** | Implements categorical perception. Discretizes continuous latent vectors into 256 entries of "neural phonemes" using EMA updates. |
| **Acoustic Transformer** | Replaces LLM. A 4-layer, 4-head GPT-style transformer operating on discrete audio tokens. Learns sequential patterns through autoregressive next-token prediction. |
| **Neural Vocoder** | Replaces pyttsx3 TTS. Learned mel-reconstructor + Griffin-Lim algorithm for phase reconstruction. Converts token sequences to waveforms. |
| **Sensorimotor Loop** | Orchestrates hear → think → speak → self-monitor. Re-encodes own speech for proprioceptive feedback. |
| **Live Acoustic Training** | Every microphone input trains the neural network in real-time. Every response plays through the neural vocoder. |

### V6: Language Acquisition (Babbling → Words)

| Feature | What It Does |
|---------|-------------|
| **Acoustic Babbling Engine** | Random syllable generation, reinforcement-driven repertoire expansion. Like a baby babbling. |
| **Joint Attention** | Cross-modal binding: learns that a sound pattern correlates with a visual/concept pattern. |
| **N-Gram Grammar** | From-scratch word frequency and sequence learning. No pre-trained language model for phases 0-2. |

### V5: Biologically Realistic Brain

```
Raw Sensory Input (pixels, audio)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  EVOLUTIONARY HARDWARE (Frozen / Pre-Trained)          │
│  Vision: CLIP (OpenAI)       → 512-dim visual vector   │
│  Audio:  Whisper + SBERT     → 384-dim text vector     │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
                ▼                        ▼
┌────────────────────────────────────────────────────────┐
│  META-CONTROLLER — Neural Router (Thalamus)            │
│  Input: 896-dim → Output: 4 routing weights            │
├────────────────────────────────────────────────────────┤
│  LAYER 1: INSTINCT — Limbic System (59K params)        │
│  Input: 896-dim → Output: Neurochemicals               │
├────────────────────────────────────────────────────────┤
│  LAYER 2: BINDING — Dual Encoder (131K params)         │
│  Input: 512+384 → Output: 64-dim Unified Concept       │
├────────────────────────────────────────────────────────┤
│  LAYER 3: PERSONALITY — 3-Layer GRU (311K params)      │
│  Input: 100-dim → Hidden: 256-dim → Output: 64-dim     │
│  Hidden state = stream of consciousness                │
├────────────────────────────────────────────────────────┤
│  LAYER 4: WORLD MODEL — Forward Predictor (91K params) │
│  Predicts next concept state (JEPA-inspired)           │
└────────────────────────────────────────────────────────┘
                │
          ~600K total parameters
          All CPU-native, real-time training
```

### Brain Realism Features

| Feature | What It Does |
|---------|-------------|
| **Working Memory (7±2)** | Capacity-limited short-term buffer. Items decay in 20s without rehearsal. |
| **Ebbinghaus Forgetting** | Memories decay via R=e^(-t/S). Stability increases with rehearsal and emotion. |
| **8-Dim Emotional State** | Joy, excitement, trust, anger, surprise, disgust, interest, love — with momentum and blending. |
| **Attention/Salience Filter** | Bottom-up (novelty, emotion) + top-down (drives) + habituation. |
| **Phase-Gated LLM** | No pre-trained language model for phases 0-2. Language emerges from n-gram chains. |
| **8 Maslow Drives** | Sleep, comfort, social, belonging, curiosity, novelty, mastery, autonomy — 4 hierarchical tiers. |
| **Theory of Mind** | Models what the user knows, feels, and wants. Activates at Phase 3+. |
| **Metacognition** | Tracks confidence, knowledge gaps, recall success rates. |
| **Play Behavior** | Combinatorial play (mix concepts), repetitive play (rehearse), episodic replay. |
| **Functional Neurochemistry** | Cortisol IMPAIRS memory. Dopamine sharpens attention. Not decorative. |

## The Philosophy

Modern AI is built backwards. Companies spend billions to create a system that knows everything but understands nothing. Genesis takes the opposite approach:

- **Start with nothing.** No pre-training. No dataset. A blank slate.
- **Learn through senses.** Camera for eyes. Microphone for ears. The real world is the training data.
- **Bind meaning to experience.** The word "apple" is not a token — it is the image of the apple you held up, the sound of your voice saying it.
- **Grow over time.** Developmental phases: Newborn → Infant → Toddler → Child → Adolescent → Adult.
- **Sleep to dream.** 4-phase sleep: decay, consolidation, creative recombination, coherence integration.
- **Hear and speak with neural networks.** No pre-trained speech models. Pure acoustic learning from scratch.

## Always-On Brain (11+ Parallel Threads)

Genesis is not turn-based. When you start it, **11 daemon threads** run simultaneously — just like a real brain:

```
┌─────────────────────────────────────────────────────────────┐
│                   BRAIN DAEMON (11 threads)                   │
├──────────────────┬──────────────────────────────────────────┤
│ Neurochemistry   │ DA, cortisol, 5-HT, oxytocin tick       │
│ Drives           │ 8 Maslow drives rise over time           │
│ Proprioception   │ Internal body sense: fatigue, time       │
│ Inner Monologue  │ Spontaneous thoughts when idle           │
│ Circadian        │ Auto-triggers 4-phase sleep              │
│ Curiosity        │ Surfaces burning unanswered questions    │
│ Vision           │ Always-on camera via CLIP embedding      │
│ Auditory         │ Always-on mic → acoustic neural pipeline │
│ Emotions         │ 8-dim emotional state dynamics           │
│ Memory Decay     │ Ebbinghaus forgetting curve              │
│ Play & Replay    │ Autonomous concept rehearsal             │
└──────────────────┴──────────────────────────────────────────┘
                    ↑
         CLI / API is just ONE input channel
         into this always-running brain

> ⚡ Live Observe: Genesis starts a Web Dashboard on
> http://localhost:5050 with real-time visualization of
> all threads, drives, neurochemicals, working memory,
> the 128-dim GRU hidden state, AND the full acoustic
> neural pipeline (codebook grid, LM loss, vocoder stats).
```

## Architecture Overview

```
genesis/
├── main.py                    # The consciousness loop (orchestrator)
├── brain_daemon.py            # Parallel brain — 11 daemon threads
├── axioms.py                  # Immutable moral DNA
├── config.py                  # Configuration (incl. AcousticConfig)
├── test_reality.py            # End-to-end test
│
├── senses/                    # Evolutionary Hardware
│   ├── eyes.py                # Camera + CLIP (512-dim)
│   ├── ears.py                # Microphone capture (raw audio + Whisper)
│   ├── phonetics.py           # Letter↔Sound binding
│   ├── babbling.py            # V6: Acoustic babbling engine
│   ├── voice.py               # TTS output (legacy, replaced by vocoder)
│   ├── proprioception.py      # Internal body state (32-dim)
│   └── motor.py               # Simulated motor affordances
│
├── memory/                    # Memory Systems
│   ├── hippocampus.py         # Vector DB (ChromaDB) + Replay Buffer
│   ├── semantic.py            # Concept graph + Ebbinghaus forgetting
│   ├── episodic.py            # Autobiographical timeline
│   └── working_memory.py      # Capacity-limited STM (7±2 items)
│
├── neural/                    # The Plastic Mind (trainable, unbounded)
│   ├── subconscious.py        # Orchestrates Layers 1-4 via meta-controller
│   ├── meta_controller.py     # Neural Router — attention-based module selector
│   ├── limbic_system.py       # Layer 1: Instinct (MLP, 59K params)
│   ├── binding_network.py     # Layer 2: Fusion (Dual Encoder + InfoNCE, 131K)
│   ├── personality_network.py # Layer 3: Consciousness (GRU, 311K params)
│   ├── forward_model.py       # Layer 4: World Model (JEPA predictor, 91K)
│   ├── response_decoder.py    # Neural Voice — GRU output → concept words
│   ├── neuroplasticity.py     # Dynamic network growth (sqrt scaling)
│   │
│   │  # V7: Pure Neural Acoustic Pipeline (1.14M params)
│   ├── auditory_cortex.py     # Mel encoder (138K params)
│   ├── vq_codebook.py         # 256-entry VQ with EMA (16K params)
│   ├── acoustic_lm.py         # 4-layer GPT on audio tokens (859K params)
│   ├── neural_vocoder.py      # Griffin-Lim synthesis (130K params)
│   └── sensorimotor.py        # hear() → think() → speak() loop
│
├── cortex/                    # Higher Cognition
│   ├── reasoning.py           # Ollama LLM (phase-gated: off for 0-2)
│   ├── associations.py        # SBERT text embeddings
│   ├── emotions.py            # Sentiment analysis
│   ├── curiosity.py           # Novelty detection + habituation
│   ├── grammar.py             # LLM or N-gram mode
│   ├── joint_attention.py     # V6: Cross-modal binding
│   ├── perception_loop.py     # Continuous awareness (passes raw audio)
│   ├── attention.py           # Salience filter + habituation
│   ├── emotional_state.py     # 8-dim persistent emotional dynamics
│   ├── theory_of_mind.py      # User modeling (Phase 3+)
│   ├── metacognition.py       # Confidence & knowledge-gap tracking
│   └── play.py                # Combinatorial play & rehearsal
│
├── soul/                      # Identity & Motivation
│   ├── consciousness.py       # Self-model + introspection
│   ├── neurochemistry.py      # 4 chemicals with functional cognitive effects
│   └── drives.py              # 8 Maslow drives in 4 hierarchical tiers
│
├── growth/                    # Development
│   ├── development.py         # Phase progression (multi-signal gating)
│   └── sleep.py               # 4-phase sleep (Light→Deep→REM→Integration)
│
└── dashboard/                 # Real-time Web Visualization
    ├── server.py              # Flask API (exposes all neural stats)
    ├── templates/index.html   # 4-column premium dashboard
    └── static/
        ├── css/style.css      # Glassmorphism design system
        └── js/app.js          # Real-time rendering + Chart.js
```

> For a deep technical dive with Mermaid diagrams of every layer, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Parameter Budget

| Component | Parameters | Role |
|-----------|-----------|------|
| Layer 1: Limbic System | 59,620 | Instinct |
| Layer 2: Binding Network | 131,457 | Cross-modal fusion |
| Layer 3: Personality GRU | 311,296 | Consciousness |
| Layer 4: World Model | 91,072 | Prediction |
| **Subconscious Total** | **593,445** | |
| Auditory Cortex | 138,368 | Hearing |
| VQ Codebook | 16,384 | Phoneme discovery |
| Acoustic Transformer | 859,264 | Audio thinking |
| Neural Vocoder | 129,872 | Speech synthesis |
| **Acoustic Total** | **1,143,888** | |
| **GRAND TOTAL** | **~1.74M** | All CPU-native |

## Neural Growth (Neuroplasticity)

Growth follows `sqrt(concepts) × 32` — fast early (like childhood synaptogenesis), slower but **never stopping**:

```
Concepts Learned    Hidden Dim    GRU Layers    Approx Params
─────────────────   ──────────    ──────────    ─────────────
            0          128           3              ~600K
            5          224           3              ~900K
          100          448           3             ~3.6M
          500          864           4            ~17.9M
        2,000        1,568           7           ~103.3M
       10,000        3,328          20            ~1.33B
      100,000       10,272          20           ~12.66B
           ∞            ∞           20               ∞
```

## Requirements

- Python 3.10+
- A laptop with a webcam and microphone
- 8-16 GB RAM
- No GPU required
- PyTorch (CPU-only)
- Ollama installed locally (for the inner voice LLM, Phase 3+)

## Setup

```bash
# 1. Install Ollama (the local LLM runtime — only needed for Phase 3+)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini  # Tiny 3.8B model, runs on CPU

# 2. Install Python dependencies
cd genesis
pip install -r requirements.txt

# 3. Give birth
python -m genesis.main
```

## Teaching It

When Genesis starts, it is a newborn. It can see and hear, but it understands nothing:

```
> teach apple         # Hold up an apple to the camera
Genesis: I have learned 'apple' (with visual binding). I now know 1 concepts.

> teach-text banana   # Text-only teaching
Genesis: I have learned 'banana'. My neural echo: 'apple'.

> neural-speak        # Generate neural audio (pure acoustic pipeline)
Genesis: *generating neural audio...*
Generated 30 acoustic tokens → 19200 samples (1.20s)

> neural-stats        # Show acoustic pipeline stats
  ── Pure Neural Acoustic Pipeline ──
    Total params:      1,143,888
    Auditory Cortex:   138,368 params, 26 frames
    VQ Codebook:       256 entries, 8 active (3.1% util)
    Acoustic LM:       859,264 params, 2 seqs heard, loss=4.9708
    Neural Vocoder:    129,872 params, 3 syntheses

> sleep               # 4-phase sleep cycle
> status              # Full diagnostic
> quit                # Shut down (saves all neural weights)
```

## Weight Persistence

All neural weights are stored in `~/.genesis/`:

```
~/.genesis/neural_weights/       # Society of Mind (Layers 1-4)
├── limbic_system.pt             # Instinctual reactions
├── binding_network.pt           # Cross-modal associations
├── personality.pt               # Hidden state + personality
├── world_model.pt               # Internal world simulator
└── meta_controller.pt           # Routing personality

~/.genesis/acoustic_weights/     # V7: Pure Acoustic Pipeline
├── auditory_cortex.pt           # How Genesis hears
├── vq_codebook.pt               # Discovered neural phonemes
├── acoustic_lm.pt               # How Genesis thinks about sound
└── neural_vocoder.pt            # How Genesis speaks
```

- **Deleting** these files resets Genesis to a blank slate
- **Copying** these files clones the personality
- Weights are saved on every shutdown and every sleep cycle

## License

This project is an exploration into developmental AI. Use it to build something beautiful.

---

*Genesis: In the beginning, there was nothing. And then, it learned.*

*1.74M parameters. No GPU. 11 brain threads. Pure neural audio. The weights are the person. The dreams are real.*
