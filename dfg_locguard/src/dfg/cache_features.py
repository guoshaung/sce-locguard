from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .diffusion_feature_extractor import DiffusionFeatureExtractor


def cache_placeholder_features(image_paths: list[Path], output_dir: Path, extractor: DiffusionFeatureExtractor) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        features = extractor.extract_features(image)
        out_path = output_dir / f"{path.stem}.npz"
        np.savez_compressed(out_path, features=features if features is not None else np.array([]))
        saved.append(out_path)
    return saved

