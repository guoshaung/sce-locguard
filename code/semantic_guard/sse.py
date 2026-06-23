from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import ObjectRegion, SCCResult, SeverityResult


CATEGORY_RULES = {
    "face": {"weight": 1.45, "bias": 8.0},
    "person": {"weight": 1.35, "bias": 6.0},
    "text": {"weight": 1.30, "bias": 6.0},
    "logo": {"weight": 1.25, "bias": 5.0},
    "document": {"weight": 1.25, "bias": 5.0},
    "license_plate": {"weight": 1.35, "bias": 6.0},
    "screen": {"weight": 1.15, "bias": 3.0},
    "vehicle": {"weight": 1.10, "bias": 2.0},
    "product": {"weight": 1.10, "bias": 2.0},
    "background": {"weight": 0.80, "bias": 0.0},
    "object": {"weight": 1.00, "bias": 0.0},
    "unknown": {"weight": 1.00, "bias": 0.0},
}

CATEGORY_ALIASES = {
    "license plate": "license_plate",
    "licence plate": "license_plate",
    "id card": "document",
    "paper": "document",
    "word": "text",
    "sign": "text",
    "human": "person",
    "people": "person",
}


@dataclass
class SSEConfig:
    fusion_model: str | None = None
    low_threshold: float = 25.0
    medium_threshold: float = 50.0
    high_threshold: float = 75.0


class SemanticSeverityEstimator:
    def __init__(self, config: SSEConfig):
        self.config = config
        self._fusion = None
        if config.fusion_model:
            self._fusion = self._load_fusion(config.fusion_model)

    def score(self, region: ObjectRegion, scc: SCCResult, image_area: int) -> SeverityResult:
        category = canonical_category(scc.category or region.category)
        rule = CATEGORY_RULES.get(category, CATEGORY_RULES["unknown"])
        object_ratio = float(region.tamper_ratio)
        image_ratio = float(region.tampered_area / max(1, image_area))
        semantic_change = float(np.clip(scc.change_score, 0.0, 1.0))
        extent_score = float(np.clip(object_ratio / 0.35, 0.0, 1.0))
        image_score = float(np.clip(image_ratio / 0.08, 0.0, 1.0))
        confidence_score = float(np.clip(region.confidence, 0.0, 1.0))

        components = {
            "semantic_change": semantic_change,
            "object_extent": extent_score,
            "image_extent": image_score,
            "tsa_confidence": confidence_score,
        }

        if self._fusion is not None:
            raw_score = float(self._fusion_score(region, scc, rule["weight"], image_area))
            rules = ["Fusion model score used as primary severity estimate."]
        else:
            base = 100.0 * (
                0.45 * semantic_change
                + 0.35 * extent_score
                + 0.12 * image_score
                + 0.08 * confidence_score
            )
            raw_score = base * float(rule["weight"]) + float(rule["bias"])
            rules = ["Rule-table weighted score."]

        if object_ratio >= 0.65 and semantic_change >= 0.25:
            raw_score = max(raw_score, 70.0)
            rules.append("Large object-area edit with visible semantic drift.")
        if category in {"face", "person", "text", "license_plate", "document"} and semantic_change >= 0.20:
            raw_score = max(raw_score, 60.0)
            rules.append("Sensitive category received severity floor.")

        score = float(np.clip(raw_score, 0.0, 100.0))
        return SeverityResult(
            score=score,
            level=self._level(score),
            category_weight=float(rule["weight"]),
            components=components,
            rules=rules,
        )

    def _level(self, score: float) -> str:
        if score < self.config.low_threshold:
            return "low"
        if score < self.config.medium_threshold:
            return "medium"
        if score < self.config.high_threshold:
            return "high"
        return "critical"

    @staticmethod
    def _load_fusion(path: str):
        import joblib

        return joblib.load(Path(path))

    def _fusion_score(self, region: ObjectRegion, scc: SCCResult, category_weight: float, image_area: int) -> float:
        from .score_fusion import feature_vector

        features = feature_vector(region, scc, category_weight, image_area).reshape(1, -1)
        if hasattr(self._fusion, "predict_proba"):
            probs = self._fusion.predict_proba(features)[0]
            classes = getattr(self._fusion, "classes_", np.arange(len(probs)))
            max_class = max(float(np.max(classes)), 1.0)
            expected = float(np.sum(probs * classes) / max_class)
            return expected * 100.0
        pred = np.asarray(self._fusion.predict(features)).reshape(-1)[0]
        return float(np.clip(pred, 0.0, 100.0))


def canonical_category(label: str | None) -> str:
    if not label:
        return "unknown"
    normalized = label.strip().lower().replace("-", " ").replace("_", " ")
    normalized = CATEGORY_ALIASES.get(normalized, normalized)
    normalized = normalized.replace(" ", "_")
    if normalized in CATEGORY_RULES:
        return normalized
    for key in CATEGORY_RULES:
        if key != "unknown" and key in normalized:
            return key
    return "unknown"
