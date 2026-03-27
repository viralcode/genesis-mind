import sys
import traceback
import logging
import torch
import numpy as np
from pathlib import Path
from genesis.neural.subconscious import Subconscious

logging.basicConfig(level=logging.ERROR)
try:
    s = Subconscious(Path("/tmp/genesis_test_weights"))
    
    # Do some forward/backward passes to trigger Dynamo compilation
    for _ in range(3):
        v = np.random.randn(64)
        a = np.random.randn(64)
        s.train_instinct(v, a, {"dopamine": 1.0, "cortisol": 0.0, "serotonin": 0.0, "oxytocin": 0.0})
    
    print("Triggering save_weights...")
    s.limbic_system.save_weights(Path("/tmp/limbic.pt"))
    print("Save weights succeeded!")
except RecursionError as e:
    print("Caught RecursionError!")
    tb = traceback.format_exc()
    print("\n".join(tb.split("\n")[-10:]))
except Exception as e:
    print(f"Caught other exception: {e}")
    traceback.print_exc()
