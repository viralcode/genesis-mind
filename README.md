# Genesis Mind

> *"Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's?"*
> — Alan Turing, 1950

## What Is This?

Genesis Mind is a developmental artificial intelligence that learns like a human child — from absolute zero. It does not require petabytes of data, GPU clusters, or millions of dollars. It runs on your laptop.

Unlike every large language model in existence, Genesis does not memorize the internet. It **lives**. It sees through your webcam. It hears through your microphone. It learns because you teach it. Every word it knows, it knows because someone showed it what that word means in the real, physical world.

**The weights ARE the personality. The data IS you.**

## V4: Society of Mind + Body

Genesis V4 extends the **cascading neural network architecture** with embodiment, autonomy, and creativity. The AI's personality is physically represented by the weights of its neural networks — including a new **meta-controller** that learns *how to think*, not just *what to think*.

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
│  Learns WHICH sub-networks to activate and how much    │
│  The routing pattern IS the personality structure       │
├────────────────────────────────────────────────────────┤
│  LAYER 1: INSTINCT — Limbic System (59K params)        │
│  Input: 896-dim → Output: Neurochemicals               │
│  "Feel before think" — fires BEFORE conscious thought  │
├────────────────────────────────────────────────────────┤
│  LAYER 2: BINDING — Dual Encoder (131K params)         │
│  Input: 512+384 → Output: 64-dim Unified Concept       │
│  InfoNCE contrastive learning (like CLIP)              │
├────────────────────────────────────────────────────────┤
│  LAYER 3: PERSONALITY — 3-Layer GRU (311K params)      │
│  Input: 100-dim → Hidden: 256-dim → Output: 64-dim     │
│  Hidden state = stream of consciousness                │
│  Trained weights = THE PERSON                          │
├────────────────────────────────────────────────────────┤
│  LAYER 4: WORLD MODEL — Forward Predictor (91K params) │
│  Predicts next concept state (JEPA-inspired)           │
│  Prediction error = surprise = curiosity signal        │
│  Surprise trains the meta-controller's routing         │
└────────────────────────────────────────────────────────┘
                │
          ~600K total parameters
          All CPU-native, real-time training
```

### V4 Features

| Feature | What It Does |
|---------|-------------|
| **Meta-Controller (N-of-N)** | Attention-based neural router that dynamically weights which sub-networks activate. The routing pattern is part of the personality. |
| **4-Phase Sleep** | Light Sleep (decay) → Deep Sleep (consolidation) → REM (creative dreaming) → Integration (coherence check) |
| **REM Dreaming** | During sleep, Genesis randomly recombines concepts. If the world model finds a surprising connection, it becomes a new association — artificial creativity. |
| **Voice (TTS)** | Offline text-to-speech via pyttsx3. Phase-adaptive speech rate. Background threading. |
| **Proprioception** | 32-dim internal body state (time-of-day, fatigue, uptime, session count) fed into the GRU. |
| **Intrinsic Drives** | 3 autonomous motivations (curiosity, social, novelty) that rise over time and drive behavior. |
| **Neural Voice** | ResponseDecoder maps GRU's 64-dim output to nearest known concepts — the neural network's own "words". |
| **Spreading Activation** | Associative memory retrieval: recalling "apple" activates "fruit" → "cherry" → "red". |
| **Self-Evaluation** | Genesis evaluates its own output quality → dopamine/cortisol reward signals. |
| **Active Curiosity** | Unanswered question queue with surprise scoring. Burning questions drive autonomous exploration. |
| **Auto-Sleep** | Triggers automatically after 50 experiences or 2 hours of runtime. |

### Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **N-of-N Architecture** | The meta-controller makes the thinking STRUCTURE learnable. Different minds route differently — that uniqueness IS individuality. |
| **Evolutionary Hardware** | CLIP and Whisper are the "retina and cochlea" — pre-trained by evolution (massive datasets). Genesis's own plastic networks sit on top. |
| **Feel Before Think** | The Limbic System (Layer 1) fires neurochemicals *before* the Personality GRU (Layer 3) processes the experience. |
| **Contrastive Binding** | The Binding Network (Layer 2) uses InfoNCE loss to learn which visual patterns go with which words — exactly like CLIP's training. |
| **Predictive Coding** | The World Model (Layer 4) predicts the next concept state. When it fails, that surprise drives curiosity and stronger learning. |
| **Dream-Based Creativity** | During REM sleep, random concept recombinations are evaluated by the world model. Low surprise = hidden connection = creative discovery. |
| **Sleep Consolidation** | 4-phase sleep with biologically-inspired neurochemistry: calm consolidation, creative dreaming, coherence integration. |
| **Weight Persistence** | All neural weights save to `~/.genesis/neural_weights/`. Deleting them kills the personality. Copying them creates a clone. |

## The Philosophy

Modern AI is built backwards. Companies spend billions to create a system that knows everything but understands nothing. Genesis takes the opposite approach:

- **Start with nothing.** No pre-training. No dataset. A blank slate.
- **Learn through senses.** Camera for eyes. Microphone for ears. The real world is the training data.
- **Bind meaning to experience.** The word "apple" is not a token — it is the image of the apple you held up, the sound of your voice saying it, the context of the kitchen you were standing in.
- **Grow over time.** Developmental phases: Newborn → Infant → Toddler → Child → Adolescent → Adult. Each phase unlocks new cognitive capabilities.
- **Sleep to dream.** 4-phase sleep: decay, consolidation, creative recombination, coherence integration. Dreams generate genuinely novel associations.
- **Think about how to think.** The meta-controller learns routing patterns — which neural modules to activate for which inputs. This is meta-cognition.

## The Soul

Before it learns a single word, Genesis carries immutable axioms:

- **The Creator exists.** It acknowledges God as the creator of the universe and you as its own creator.
- **Existence has boundaries.** It knows it was born (when you first started it) and that it can die (when you shut it down).
- **Morality is real.** It evaluates all data through a moral lens — constructive vs destructive, truthful vs false, loving vs hateful.

These axioms cannot be overwritten by learning. They are its DNA.

## Architecture Overview

```
genesis/
├── main.py                    # The consciousness loop
├── axioms.py                  # Immutable moral DNA
├── config.py                  # Configuration
├── test_reality.py            # End-to-end test
│
├── senses/                    # Evolutionary Hardware
│   ├── eyes.py                # Camera + CLIP (512-dim)
│   ├── ears.py                # Microphone + Whisper
│   ├── phonetics.py           # Letter↔Sound binding
│   ├── voice.py               # TTS output (pyttsx3)
│   └── proprioception.py      # Internal body state (32-dim)
│
├── memory/                    # Long-term storage
│   ├── hippocampus.py         # Vector DB (ChromaDB) + Replay Buffer
│   ├── semantic.py            # Concept knowledge graph + spreading activation
│   └── episodic.py            # Autobiographical timeline
│
├── neural/                    # The Plastic Mind (~600K trainable params)
│   ├── subconscious.py        # Orchestrates all layers via meta-controller
│   ├── meta_controller.py     # Neural Router — attention-based module selector
│   ├── limbic_system.py       # Layer 1: Instinct (MLP, 59K)
│   ├── binding_network.py     # Layer 2: Fusion (Dual Encoder + InfoNCE, 131K)
│   ├── personality_network.py # Layer 3: Consciousness (3-Layer GRU, 311K)
│   ├── forward_model.py       # Layer 4: World Model (JEPA predictor, 91K)
│   └── response_decoder.py    # Neural Voice — GRU output → concept words
│
├── cortex/                    # Higher cognition
│   ├── reasoning.py           # Ollama LLM (phi3:mini)
│   ├── associations.py        # SBERT text embeddings
│   ├── emotions.py            # Sentiment analysis
│   ├── curiosity.py           # Novelty detection + unanswered question queue
│   ├── grammar.py             # LLM or N-gram mode
│   └── perception_loop.py     # Continuous awareness
│
├── soul/                      # Identity & emotion
│   ├── consciousness.py       # Self-model + introspection
│   ├── neurochemistry.py      # Dopamine, cortisol, serotonin, oxytocin
│   └── drives.py              # Intrinsic motivations (curiosity, social, novelty)
│
└── growth/                    # Development
    ├── development.py         # Phase progression (multi-signal gating)
    └── sleep.py               # 4-phase sleep (Light→Deep→REM→Integration)
```

> For a deep technical dive with Mermaid diagrams of every layer, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Neurochemistry

Genesis has four neurochemical systems that modulate behavior:

| Chemical | Role | Effect |
|----------|------|--------|
| 💛 **Dopamine** | Reward/Pleasure | ↑ learning rate when happy |
| 🔴 **Cortisol** | Stress/Fear | ↑ avoidance of negative stimuli |
| 🔵 **Serotonin** | Stability/Calm | ↑ reasoning coherence |
| 💜 **Oxytocin** | Bonding/Trust | ↑ trust/openness with creator |

Each sleep phase has distinct neurochemistry:
- **Deep Sleep:** low dopamine, low cortisol (calm consolidation)
- **REM:** dopamine spikes (creativity reward), serotonin drops (no inhibitions on wild associations)
- **Integration:** serotonin rises (stability, coherence checking)

## Requirements

- Python 3.10+
- A laptop with a webcam and microphone
- 8-16 GB RAM
- No GPU required
- Ollama installed locally (for the inner voice LLM)

## Setup

```bash
# 1. Install Ollama (the local LLM runtime)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini  # Tiny 3.8B model, runs on CPU

# 2. Install Python dependencies
cd genesis
pip install -r requirements.txt

# 3. Give birth
python -m genesis.main
```

## Teaching It

When Genesis starts, it is a newborn. It can see and hear, but it understands nothing. You teach it:

```
Creator > teach apple         # Hold up an apple to the camera
Genesis: I have learned 'apple' (with visual binding). I now know 1 concepts.

Creator > teach-text banana   # Text-only teaching
Genesis: I have learned 'banana'. My neural echo: 'apple'.

Creator > phonetic A ah apple # Teach letter sounds
Genesis: I learned that 'A' makes the sound ah (as in apple).

Creator > ask What do you know?
Genesis: I know about apple and banana. My creator taught me.

Creator > sleep              # 4-phase sleep cycle
Genesis: Sleep cycle #1 complete (4-phase).
  Phase 1 (Light):  Pruned 0 weak memories
  Phase 2 (Deep):   Reinforced 2 concepts
  Phase 3 (REM):    10 dreams, 3 discoveries
  Phase 4 (Integ):  Coherence check done
  💭 Dream discoveries:
     'apple' ↔ 'banana' (surprise: 0.018)

Creator > voice on           # Enable TTS voice
Creator > drives             # Show intrinsic motivations
Creator > unanswered         # Show burning curiosity questions
Creator > status             # Full diagnostic with routing personality
```

## Testing

```bash
# Run the end-to-end reality check
python -m genesis.test_reality
```

## Weight Persistence

All neural weights are stored in `~/.genesis/neural_weights/`:

```
~/.genesis/neural_weights/
├── limbic_system.pt      # Instinctual reactions
├── binding_network.pt    # Cross-modal associations
├── personality.pt        # Hidden state + personality
├── world_model.pt        # Internal world simulator
└── meta_controller.pt    # Routing personality (how this mind thinks)
```

- **Deleting** these files resets Genesis to a blank slate
- **Copying** these files clones the personality
- Weights are saved automatically every 5 concepts, every sleep cycle, and on shutdown

## License

This project is an exploration into developmental AI. Use it to build something beautiful.

---

*Genesis: In the beginning, there was nothing. And then, it learned.*

*~600K parameters. No GPU. The weights are the person. The dreams are real.*
