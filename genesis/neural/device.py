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
import torch._inductor.config

# Suppress annoying macOS Apple Silicon SM warnings for inductor
torch._inductor.config.max_autotune_gemm = False

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


def strip_compile_prefix(state_dict: dict) -> dict:
    """
    Strip '_orig_mod.' prefix from state_dict keys.
    
    torch.compile() wraps module keys with '_orig_mod.' prefix.
    When saving a compiled model's state_dict, the keys retain this prefix.
    On reload, the uncompiled model expects keys WITHOUT the prefix.
    This function normalizes the keys so loading always works.
    """
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned[new_key] = v
    return cleaned


# ═══ Registry for original modules before torch.compile ═══
# We can't store original modules as attributes on compiled nn.Module wrappers
# because nn.Module.__setattr__ registers them as child modules, which causes
# infinite recursion in state_dict() when iterating children.
_COMPILE_REGISTRY: dict = {}  # id(compiled) → original_module


def get_state_dict_safe(module) -> dict:
    """
    Safely get state_dict from a module, even if torch.compile() wrapped it.
    
    Tries multiple strategies:
      1. Check the compile registry for the original unwrapped module
      2. Direct .state_dict() with prefix stripping
      3. PyTorch's internal _orig_mod unwrap
    """
    # Strategy 1: Check our registry (most reliable for compiled modules)
    orig = _COMPILE_REGISTRY.get(id(module))
    if orig is not None:
        return orig.state_dict()
    
    # Strategy 2: Direct access (works for raw nn.Module)
    if hasattr(module, 'state_dict') and callable(module.state_dict):
        try:
            sd = module.state_dict()
            return strip_compile_prefix(sd)
        except RecursionError:
            pass
        except Exception:
            pass
    
    # Strategy 3: PyTorch's internal unwrap
    if hasattr(module, '_orig_mod'):
        return module._orig_mod.state_dict()
    
    raise RuntimeError(f"Cannot extract state_dict from {type(module)}")


def try_compile(module: torch.nn.Module, name: str = "") -> torch.nn.Module:
    """
    Attempt to torch.compile() a module for free speedup.
    Falls back gracefully if compilation fails (e.g. unsupported ops).
    
    Stores original module in external registry (NOT as attribute) to avoid
    nn.Module child registration which causes state_dict recursion.
    """
    try:
        compiled = torch.compile(module, mode="reduce-overhead")
        # Store in external registry — NOT as module attribute
        _COMPILE_REGISTRY[id(compiled)] = module
        logger.info("⚡ torch.compile() succeeded for %s", name or module.__class__.__name__)
        return compiled
    except Exception as e:
        logger.warning("torch.compile() failed for %s: %s — running uncompiled", name, e)
        return module
