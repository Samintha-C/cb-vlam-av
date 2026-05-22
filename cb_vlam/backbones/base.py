"""Abstract backbone interface for feature extraction.

A backbone is a frozen VLA model that produces fixed-size feature vectors
from (image, prompt) pairs. These features feed the Concept Bottleneck Layer.
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from PIL import Image


class BaseBackbone(ABC):
    """Interface every backbone must implement."""

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load model weights, processor, etc. onto the given device."""

    @abstractmethod
    def extract(self, image: Image.Image, user_prompt: str) -> Dict[str, np.ndarray]:
        """Forward pass; return one or more feature vectors per sample.

        Returns:
            Dict mapping a feature name to a (D,) float32 numpy array.
            Feature names are backbone-specific.
        """
