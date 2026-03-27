"""
Genesis Mind — The Eyes (V8: No Pretrained Models)

The visual perception system. Opens the webcam, captures frames,
detects meaningful changes, and produces embeddings using the
from-scratch VisualCortex (Conv2D autoencoder, ~50K params).

V8 CHANGE: Removed CLIP ViT-B/32 (150M+ pretrained params).
Vision is now entirely from-scratch. Genesis starts blind and
learns to see through self-supervised reconstruction learning.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger("genesis.senses.eyes")


@dataclass
class VisualPercept:
    """A single moment of seeing."""
    image: np.ndarray                       # Raw image (H, W, 3) BGR
    embedding: Optional[np.ndarray] = None  # Visual cortex embedding (64-dim)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    motion_score: float = 0.0
    is_significant: bool = False


class Eyes:
    """
    The visual perception system of Genesis.

    Opens the laptop's webcam and observes the world. Generates
    embeddings using the from-scratch VisualCortex — NOT a pretrained
    model. Genesis starts blind and learns to see.
    """

    def __init__(self, camera_index: int = 0, image_size: Tuple[int, int] = (64, 64),
                 motion_threshold: float = 0.05, visual_cortex=None):
        self.camera_index = camera_index
        self.image_size = image_size
        self.motion_threshold = motion_threshold
        self._visual_cortex = visual_cortex  # Injected from GenesisMind

        self._camera = None
        self._last_frame = None

        logger.info("Eyes initialized (camera_index=%d, size=%s)", camera_index, image_size)

    def set_visual_cortex(self, cortex):
        """Set the visual cortex reference (for late binding)."""
        self._visual_cortex = cortex

    def open(self):
        """Open the eyes — activate the webcam."""
        import cv2

        self._camera = cv2.VideoCapture(self.camera_index)
        if not self._camera.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}. "
                "Ensure a webcam is connected and accessible."
            )
        logger.info("Eyes opened — camera %d is active", self.camera_index)

    def close(self):
        """Close the eyes — release the webcam."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None
            logger.info("Eyes closed")

    def _compute_motion(self, frame: np.ndarray) -> float:
        """Compute how much the visual field has changed since last frame."""
        if self._last_frame is None:
            return 1.0

        import cv2
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_previous = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        diff = np.mean(np.abs(gray_current - gray_previous))
        return float(diff)

    def look(self) -> Optional[VisualPercept]:
        """Take a single look at the world."""
        import cv2

        if self._camera is None:
            self.open()

        ret, frame = self._camera.read()
        if not ret:
            logger.warning("Failed to capture frame from camera")
            return None

        frame_resized = cv2.resize(frame, self.image_size)

        motion = self._compute_motion(frame_resized)
        is_significant = motion > self.motion_threshold

        percept = VisualPercept(
            image=frame_resized,
            motion_score=motion,
            is_significant=is_significant,
        )

        self._last_frame = frame_resized.copy()
        self._last_frame_full = frame.copy()  # Full-res for dashboard (must copy or buffer is lost)
        return percept

    def embed(self, percept: VisualPercept, train: bool = True) -> np.ndarray:
        """
        Convert what the eyes see into a 64-dim vector using the
        from-scratch VisualCortex (Conv2D autoencoder).

        Unlike CLIP (which was pretrained on billions of image-text pairs),
        this embedding is learned from scratch through self-supervised
        reconstruction. Initially random — improves with experience.
        """
        if self._visual_cortex is None:
            logger.warning("No visual cortex connected — returning zero embedding")
            return np.zeros(64, dtype=np.float32)

        # Convert BGR → RGB for the cortex
        image_rgb = percept.image[:, :, ::-1].copy()  # BGR -> RGB
        embedding = self._visual_cortex.see(image_rgb, train=train)
        percept.embedding = embedding
        return embedding

    def embed_image(self, image: np.ndarray, train: bool = True) -> np.ndarray:
        """Embed a raw image array (for teaching without VisualPercept)."""
        if self._visual_cortex is None:
            return np.zeros(64, dtype=np.float32)
        
        if image.shape[2] == 3 and len(image.shape) == 3:
            image_rgb = image[:, :, ::-1].copy()  # Assume BGR → RGB
        else:
            image_rgb = image
        return self._visual_cortex.see(image_rgb, train=train)

    def show_preview(self, duration_sec: float = 10.0):
        """Show a live preview of what the eyes see."""
        import cv2

        if self._camera is None:
            self.open()

        start = time.time()
        logger.info("Showing camera preview for %.1f seconds...", duration_sec)

        while time.time() - start < duration_sec:
            ret, frame = self._camera.read()
            if not ret:
                break
            cv2.imshow("Genesis Eyes — Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    from genesis.neural.visual_cortex import VisualCortex

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Eyes Test (V8: From-Scratch Vision)")
    print("=" * 60)

    cortex = VisualCortex()
    with Eyes(visual_cortex=cortex) as eyes:
        eyes.show_preview(duration_sec=5.0)
        print("\nTaking a snapshot...")
        percept = eyes.look()
        if percept:
            embedding = eyes.embed(percept)
            print(f"  Image shape: {percept.image.shape}")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
            print(f"  Motion score: {percept.motion_score:.4f}")
            print(f"  Visual cortex stats: {cortex.get_stats()}")
            print("Eyes test PASSED ✓")
