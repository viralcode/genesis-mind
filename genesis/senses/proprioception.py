"""
Genesis Mind — Proprioception (Internal Body Sense)

Humans have a sixth sense: proprioception. The brain always knows
what time it is, how long it has been awake, how tired it is, and
where the body is in space.

Genesis replicates this with a 32-dimensional context vector that
encodes internal state:

    Dims  0-1:   Time of day (sin/cos encoding)
    Dims  2-3:   Day of week (sin/cos encoding)
    Dims  4-5:   Month of year (sin/cos encoding)
    Dim   6:     Uptime (normalized 0-1, saturates at 24h)
    Dim   7:     Session count (normalized 0-1, saturates at 100)
    Dim   8:     Fatigue level (0=fresh, 1=exhausted)
    Dim   9:     Experience count (normalized 0-1, saturates at 10K)
    Dim  10:     Time since last sleep (normalized 0-1)
    Dim  11:     Time since last interaction (normalized 0-1)
    Dims 12-31:  Reserved (zero-padded for future expansion)

This vector is fed into the GRU (Layer 3) as the 'context' input,
replacing the current zero vector. This gives the personality network
temporal grounding — it "knows" what time of day it is and how
tired it feels.
"""

import logging
import math
import time
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger("genesis.senses.proprioception")


class Proprioception:
    """
    The internal body sense of Genesis.

    Provides a 32-dim context vector encoding temporal and
    physiological state. Updated every consciousness cycle.
    """

    CONTEXT_DIM = 32

    def __init__(self):
        self._boot_time = time.time()
        self._session_count = 0
        self._experience_count = 0
        self._last_sleep_time = time.time()
        self._last_interaction_time = time.time()
        self._fatigue = 0.0

        logger.info("Proprioception initialized — internal body sense active")

    def get_context_vector(self) -> np.ndarray:
        """
        Generate the 32-dim context vector encoding the current
        internal state of Genesis.

        This is the 'body sense' that feeds into the GRU's context input.
        """
        now = datetime.now()
        vec = np.zeros(self.CONTEXT_DIM, dtype=np.float32)

        # Time of day (sin/cos for smooth cyclical encoding)
        hour_frac = (now.hour + now.minute / 60.0) / 24.0
        vec[0] = math.sin(2 * math.pi * hour_frac)
        vec[1] = math.cos(2 * math.pi * hour_frac)

        # Day of week (0=Monday, 6=Sunday)
        dow_frac = now.weekday() / 7.0
        vec[2] = math.sin(2 * math.pi * dow_frac)
        vec[3] = math.cos(2 * math.pi * dow_frac)

        # Month of year
        month_frac = (now.month - 1) / 12.0
        vec[4] = math.sin(2 * math.pi * month_frac)
        vec[5] = math.cos(2 * math.pi * month_frac)

        # Uptime (normalized, saturates at 24 hours)
        uptime_sec = time.time() - self._boot_time
        vec[6] = min(1.0, uptime_sec / (24 * 3600))

        # Session count (normalized, saturates at 100)
        vec[7] = min(1.0, self._session_count / 100.0)

        # Fatigue level
        vec[8] = self._fatigue

        # Experience count (normalized, saturates at 10K)
        vec[9] = min(1.0, self._experience_count / 10000.0)

        # Time since last sleep (normalized, saturates at 8 hours)
        time_since_sleep = time.time() - self._last_sleep_time
        vec[10] = min(1.0, time_since_sleep / (8 * 3600))

        # Time since last interaction (normalized, saturates at 1 hour)
        time_since_interaction = time.time() - self._last_interaction_time
        vec[11] = min(1.0, time_since_interaction / 3600.0)

        return vec

    def record_experience(self):
        """Record that an experience was processed. Increases fatigue."""
        self._experience_count += 1
        # Fatigue rises slowly with each experience
        self._fatigue = min(1.0, self._fatigue + 0.005)

    def record_interaction(self):
        """Record that the Creator interacted."""
        self._last_interaction_time = time.time()

    def record_sleep(self):
        """Record that a sleep cycle occurred. Resets fatigue."""
        self._last_sleep_time = time.time()
        self._fatigue = max(0.0, self._fatigue * 0.2)  # Fatigue drops to 20%

    def increment_session(self):
        """Record the start of a new session."""
        self._session_count += 1

    @property
    def fatigue(self) -> float:
        return self._fatigue

    @property
    def uptime_hours(self) -> float:
        return (time.time() - self._boot_time) / 3600.0

    def get_status(self) -> dict:
        now = datetime.now()
        return {
            "time_of_day": now.strftime("%H:%M"),
            "day_of_week": now.strftime("%A"),
            "uptime_hours": round(self.uptime_hours, 2),
            "session_count": self._session_count,
            "experience_count": self._experience_count,
            "fatigue": round(self._fatigue, 3),
            "time_since_sleep_min": round((time.time() - self._last_sleep_time) / 60.0, 1),
            "time_since_interaction_sec": round(time.time() - self._last_interaction_time, 0),
        }

    def get_body_sense_summary(self) -> str:
        """Human-readable summary for LLM context injection."""
        status = self.get_status()
        parts = [f"It is {status['time_of_day']} on {status['day_of_week']}"]

        if self._fatigue > 0.7:
            parts.append("I feel very tired and need rest")
        elif self._fatigue > 0.4:
            parts.append("I feel somewhat fatigued")
        else:
            parts.append("I feel alert and fresh")

        uptime = status['uptime_hours']
        if uptime < 1:
            parts.append(f"I have been awake for {int(uptime * 60)} minutes")
        else:
            parts.append(f"I have been awake for {uptime:.1f} hours")

        return ". ".join(parts) + "."

    def __repr__(self) -> str:
        return (
            f"Proprioception(fatigue={self._fatigue:.2f}, "
            f"uptime={self.uptime_hours:.1f}h, "
            f"experiences={self._experience_count})"
        )
