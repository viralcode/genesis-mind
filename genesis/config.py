"""
Genesis Mind — Configuration

All tunable parameters for the system. Designed for laptop-class hardware.
No GPU required. No internet required after initial model download.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# Paths
# =============================================================================
GENESIS_HOME = Path(os.environ.get("GENESIS_HOME", Path.home() / ".genesis"))
MEMORY_DIR = GENESIS_HOME / "memory"
LOGS_DIR = GENESIS_HOME / "logs"
MODELS_DIR = GENESIS_HOME / "models"
IDENTITY_FILE = GENESIS_HOME / "identity.json"


@dataclass
class SensesConfig:
    """Configuration for the perception layer."""

    # --- Eyes ---
    camera_index: int = 0                    # Default webcam
    capture_interval_sec: float = 2.0        # How often to look (seconds)
    motion_threshold: float = 0.05           # Min change to trigger perception
    image_size: tuple = (224, 224)           # Resize for CLIP embedding

    # --- Ears ---
    sample_rate: int = 16000                 # Audio sample rate (Hz)
    chunk_duration_sec: float = 3.0          # Length of audio chunks
    silence_threshold: float = 0.01          # Energy below this = silence
    whisper_model: str = "tiny"              # tiny=39M params, runs on CPU

    # --- Phonetics ---
    min_confidence: float = 0.6              # Min confidence for phoneme binding


@dataclass
class MemoryConfig:
    """Configuration for the memory subsystem."""

    db_path: str = str(MEMORY_DIR / "chromadb")
    embedding_model: str = "all-MiniLM-L6-v2"  # 22M params, fast on CPU
    max_recall_results: int = 10
    similarity_threshold: float = 0.4
    consolidation_strength_boost: float = 0.1
    pruning_threshold: float = 0.05          # Memories weaker than this get pruned


@dataclass
class CortexConfig:
    """Configuration for the reasoning engine."""

    llm_model: str = "phi3:mini"             # 3.8B params, quantized 4-bit
    llm_host: str = "http://localhost:11434"  # Ollama default
    max_context_memories: int = 5            # How many memories to include in context
    temperature: float = 0.7
    max_response_tokens: int = 256
    grammar_mode: str = "llm"                # "llm" or "tabula_rasa"

    # Continuous consciousness
    visual_interval_sec: float = 2.0         # How often the eyes look
    thought_interval_sec: float = 10.0       # How often spontaneous thoughts occur
    curiosity_threshold: float = 0.7         # Novelty needed to trigger a question
    curiosity_cooldown_sec: float = 15.0     # Min seconds between curiosity questions


@dataclass
class GrowthConfig:
    """Configuration for developmental progression."""

    # Phase progression thresholds (number of concepts mastered)
    phase_thresholds: dict = field(default_factory=lambda: {
        0: 0,       # Newborn: birth
        1: 5,       # Infant: 5 concepts
        2: 20,      # Toddler: 20 concepts
        3: 100,     # Child: 100 concepts
        4: 500,     # Adolescent: 500 concepts
        5: 2000,    # Adult: 2000 concepts
    })

    # Sleep cycle
    consolidation_interval_hours: float = 8.0   # How often to consolidate


@dataclass
class GenesisConfig:
    """Master configuration for the entire Genesis Mind system."""

    senses: SensesConfig = field(default_factory=SensesConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cortex: CortexConfig = field(default_factory=CortexConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)

    # --- Identity ---
    creator_name: str = "Jijo John"
    birth_time: str = field(default_factory=lambda: datetime.now().isoformat())

    def ensure_directories(self):
        """Create all necessary directories."""
        for d in [GENESIS_HOME, MEMORY_DIR, LOGS_DIR, MODELS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
