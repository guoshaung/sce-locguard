from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


Box = tuple[int, int, int, int]


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


@dataclass
class Region:
    region_id: int
    mask: np.ndarray = field(repr=False)
    bbox: Box
    area: int
    label: str = "unknown"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bbox": list(self.bbox),
            "area": int(self.area),
            "label": self.label,
            "confidence": float(self.confidence),
        }


@dataclass
class TamperRegion:
    region: Region
    tamper_area: int
    tamper_ratio: float
    region_iou: float
    mask_confidence: float
    object_tamper_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = self.region.to_dict()
        payload.update(
            {
                "tamper_area": int(self.tamper_area),
                "tamper_ratio": float(self.tamper_ratio),
                "region_iou": float(self.region_iou),
                "mask_confidence": float(self.mask_confidence),
                "object_tamper_score": float(self.object_tamper_score),
            }
        )
        return payload


@dataclass
class SemanticResult:
    semantic_similarity: float
    semantic_change_score: float
    feature_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_similarity": float(self.semantic_similarity),
            "semantic_change_score": float(self.semantic_change_score),
            "feature_method": self.feature_method,
        }


@dataclass
class SeverityResult:
    severity_score: float
    severity_level: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity_score": float(self.severity_score),
            "severity_level": self.severity_level,
            "reason": self.reason,
        }


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str | Path, size: tuple[int, int] | None = None, threshold: float = 0.5) -> np.ndarray:
    arr = load_mask_float(path, size=size)
    return arr > threshold


def load_mask_float(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.astype(np.float32)
        arr = _normalize_mask_array(arr)
        if size is not None and (arr.shape[1], arr.shape[0]) != size:
            image = Image.fromarray((arr * 255).astype(np.uint8)).resize(size, NEAREST)
            arr = np.asarray(image, dtype=np.float32) / 255.0
        return arr

    image = Image.open(path)
    if image.mode not in {"L", "I", "F"}:
        image = image.convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, NEAREST)
    arr = np.asarray(image, dtype=np.float32)
    return _normalize_mask_array(arr)


def _normalize_mask_array(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    max_v = float(arr.max()) if arr.size else 0.0
    min_v = float(arr.min()) if arr.size else 0.0
    if max_v <= 1.0 and min_v >= 0.0:
        return arr
    if max_v <= 255.0 and min_v >= 0.0:
        return arr / 255.0
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def bbox_from_mask(mask: np.ndarray) -> Box:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_box(box: Box, width: int, height: int, padding: int) -> Box:
    x1, y1, x2, y2 = box
    return max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding)


def resolve_path(path: str | Path | None, root: Path) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path
