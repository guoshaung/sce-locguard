from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def semantic_difference_map(
    original_features,
    tampered_features,
    image_size: tuple[int, int],
    fallback_original: Image.Image | None = None,
    fallback_tampered: Image.Image | None = None,
) -> np.ndarray:
    """Build a normalized semantic guidance map.

    If diffusion features are unavailable, use a pixel-difference placeholder.
    """

    if original_features is not None and tampered_features is not None:
        diff = _feature_difference(original_features, tampered_features)
        return _resize_normalize(diff, image_size)

    if fallback_original is None or fallback_tampered is None:
        return np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    arr_o = np.asarray(fallback_original.convert("RGB").resize(image_size), dtype=np.float32) / 255.0
    arr_t = np.asarray(fallback_tampered.convert("RGB").resize(image_size), dtype=np.float32) / 255.0
    diff = np.mean(np.abs(arr_o - arr_t), axis=2)
    return _resize_normalize(diff, image_size)


def _feature_difference(original_features, tampered_features) -> np.ndarray:
    f_o = np.asarray(original_features, dtype=np.float32)
    f_t = np.asarray(tampered_features, dtype=np.float32)
    f_o = f_o / np.maximum(1e-8, np.linalg.norm(f_o, axis=0, keepdims=True))
    f_t = f_t / np.maximum(1e-8, np.linalg.norm(f_t, axis=0, keepdims=True))
    return np.linalg.norm(f_o - f_t, axis=0)


def _resize_normalize(diff: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    diff = cv2.resize(diff.astype(np.float32), image_size, interpolation=cv2.INTER_LINEAR)
    min_v = float(diff.min())
    max_v = float(diff.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(diff, dtype=np.float32)
    return ((diff - min_v) / (max_v - min_v)).astype(np.float32)

