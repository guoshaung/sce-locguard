from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class DiffusionFeatureConfig:
    model_name: str = "stable-diffusion-placeholder"
    device: str = "cpu"
    enabled: bool = False


class DiffusionFeatureExtractor:
    """Placeholder interface for frozen diffusion feature extraction.

    Version 0 intentionally does not require Stable Diffusion / DIFT.
    Later this class can return UNet intermediate feature maps.
    """

    def __init__(self, model_name: str = "stable-diffusion-placeholder", device: str = "cpu", enabled: bool = False):
        self.config = DiffusionFeatureConfig(model_name=model_name, device=device, enabled=enabled)

    def extract_features(self, image: Image.Image | np.ndarray):
        if not self.config.enabled:
            return None
        raise NotImplementedError(
            "Diffusion feature extraction is reserved for future training-time semantic guidance."
        )

