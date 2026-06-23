from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
from PIL import Image


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def ensure_size(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, BICUBIC)


def load_mask(
    path: str | Path,
    target_size: Tuple[int, int] | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    mask_image = Image.open(path).convert("L")
    if target_size is not None and mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, NEAREST)
    mask = np.asarray(mask_image, dtype=np.float32) / 255.0
    return mask > threshold


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_to_uint8(mask)).save(path)


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_label_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
