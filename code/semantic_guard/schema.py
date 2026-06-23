from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np


Box = Tuple[int, int, int, int]


def bbox_from_mask(mask: np.ndarray) -> Box:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def pad_box(box: Box, width: int, height: int, padding: int) -> Box:
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


@dataclass
class ObjectRegion:
    region_id: int
    object_mask: np.ndarray = field(repr=False)
    edit_mask: np.ndarray = field(repr=False)
    bbox: Box
    source: str
    confidence: float
    category: str = "unknown"
    category_confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def object_area(self) -> int:
        return int(self.object_mask.sum())

    @property
    def tampered_area(self) -> int:
        return int((self.object_mask & self.edit_mask).sum())

    @property
    def tamper_ratio(self) -> float:
        if self.object_area == 0:
            return 0.0
        return float(self.tampered_area / self.object_area)

    def to_dict(self, image_area: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.region_id,
            "source": self.source,
            "bbox_xyxy": list(self.bbox),
            "object_area": self.object_area,
            "tampered_area": self.tampered_area,
            "object_tamper_ratio": self.tamper_ratio,
            "tsa_confidence": float(self.confidence),
            "category": self.category,
            "category_confidence": float(self.category_confidence),
            "notes": self.notes,
        }
        if image_area:
            payload["image_tamper_ratio"] = float(self.tampered_area / image_area)
        return payload


@dataclass
class SCCResult:
    backend: str
    similarity: float
    change_score: float
    category: str = "unknown"
    category_confidence: float = 0.0
    pixel_delta: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "similarity": float(self.similarity),
            "change_score": float(self.change_score),
            "category": self.category,
            "category_confidence": float(self.category_confidence),
            "pixel_delta": float(self.pixel_delta),
            "components": {k: float(v) for k, v in self.components.items()},
            "notes": self.notes,
        }


@dataclass
class SeverityResult:
    score: float
    level: str
    category_weight: float
    components: dict[str, float]
    rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "level": self.level,
            "category_weight": float(self.category_weight),
            "components": {k: float(v) for k, v in self.components.items()},
            "rules": self.rules,
        }
