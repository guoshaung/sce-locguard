from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import SemanticResult, SeverityResult, TamperRegion


HIGH_RISK = {"face", "person", "text", "logo", "license_plate", "license plate", "document", "id", "signature"}
MEDIUM_RISK = {"vehicle", "product", "object", "animal", "building"}
LOW_RISK = {"sky", "wall", "grass", "road", "background", "texture"}


@dataclass
class SSEConfig:
    low_threshold: float = 25.0
    medium_threshold: float = 50.0
    high_threshold: float = 75.0


class SeverityEstimator:
    """SSE v0: rule-based semantic severity estimation."""

    def __init__(self, config: SSEConfig):
        self.config = config

    def estimate(self, tamper_region: TamperRegion, semantic: SemanticResult, label: str | None = None) -> SeverityResult:
        label = canonical_label(label or tamper_region.region.label)
        tamper_ratio = float(tamper_region.tamper_ratio)
        semantic_change = float(semantic.semantic_change_score)
        base = 100.0 * (
            0.45 * np.clip(semantic_change / 0.5, 0.0, 1.0)
            + 0.35 * np.clip(tamper_ratio / 0.4, 0.0, 1.0)
            + 0.20 * np.clip(tamper_region.object_tamper_score, 0.0, 1.0)
        )

        if label in HIGH_RISK:
            base *= 1.25
            reason = f"high-risk label '{label}' with tamper_ratio={tamper_ratio:.2f}"
            if tamper_ratio > 0.3 and semantic_change > 0.4:
                return SeverityResult(95.0, "critical", reason + " and strong semantic change")
            base = max(base, 60.0)
        elif label in MEDIUM_RISK:
            base *= 1.05
            reason = f"medium-risk label '{label}'"
        elif label in LOW_RISK:
            base *= 0.75
            reason = f"low-risk label '{label}'"
            if semantic_change < 0.2:
                base = min(base, 30.0)
        else:
            reason = "unknown label; severity estimated from tamper extent and semantic change"

        score = float(np.clip(base, 0.0, 100.0))
        return SeverityResult(score, self._level(score), reason)

    def _level(self, score: float) -> str:
        if score < self.config.low_threshold:
            return "low"
        if score < self.config.medium_threshold:
            return "medium"
        if score < self.config.high_threshold:
            return "high"
        return "critical"


def canonical_label(label: str | None) -> str:
    if not label:
        return "unknown"
    value = label.strip().lower().replace("-", " ").replace("_", " ")
    if value == "license plate":
        return "license_plate"
    return value.replace(" ", "_")

