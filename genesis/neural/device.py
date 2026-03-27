"""
Genesis Mind — Device Management + Performance

Central device detection for hardware acceleration.
Priority: MPS (Apple Silicon GPU) → CUDA → CPU

Also provides:
    - torch.compile() for graph-level speedup
    - AMP (Automatic Mixed Precision) for 2x throughput
    - Device-aware tensor utilities

All neural modules import from here for consistent acceleration.
"""

import logging
from contextlib import nullcontext

import torch

logger = logging.getLogger("genesis.neural.device")


def _detect_device() -> torch.device:
    """Detect the best available device."""
    if torch.backends.mps.is_available():
        logger.info("🚀 MPS (Apple Silicon GPU) detected — using hardware acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("🚀 CUDA GPU detected — using hardware acceleration")
        return torch.device("cuda")
    else:
        logger.info("💻 No GPU detected — using CPU")
        return torch.device("cpu")


DEVICE = _detect_device()


def _detect_amp_config():
    """Detect AMP dtype and device_type for autocast."""
    if DEVICE.type == "cuda":
        return "cuda", torch.float16
    elif DEVICE.type == "mps":
        # MPS supports bfloat16 autocast on newer PyTorch
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.bfloat16


AMP_DEVICE_TYPE, AMP_DTYPE = _detect_amp_config()


def to_device(tensor_or_module):
    """Move a tensor or nn.Module to the global device."""
    return tensor_or_module.to(DEVICE)


def get_autocast_context():
    """
    Get the appropriate AMP autocast context manager.
    Returns nullcontext() if AMP is not beneficial.
    """
    if DEVICE.type in ("cuda", "mps"):
        return torch.autocast(device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE)
    return nullcontext()


def try_compile(module: torch.nn.Module, name: str = "") -> torch.nn.Module:
    """
    Attempt to torch.compile() a module for free speedup.
    Falls back gracefully if compilation fails (e.g. unsupported ops).
    """
    try:
        compiled = torch.compile(module, mode="reduce-overhead")
        logger.info("⚡ torch.compile() succeeded for %s", name or module.__class__.__name__)
        return compiled
    except Exception as e:
        logger.warning("torch.compile() failed for %s: %s — running uncompiled", name, e)
        return module
