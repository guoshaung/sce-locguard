from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import Region, TamperRegion


@dataclass
class TSAConfig:
    tamper_ratio_weight: float = 0.55
    region_iou_weight: float = 0.30
    confidence_weight: float = 0.15
    select_threshold: float = 0.05


class TamperToSemanticAggregator:
    """TSA v0: aggregate pixel tamper evidence to candidate semantic regions."""

    def __init__(self, config: TSAConfig):
        self.config = config

    def aggregate(
        self,
        tamper_mask: np.ndarray,
        regions: list[Region],
        confidence_map: np.ndarray | None = None,
    ) -> list[TamperRegion]:
        candidates: list[TamperRegion] = []
        for region in regions:
            inter = region.mask & tamper_mask
            tamper_area = int(inter.sum())
            region_area = max(1, int(region.mask.sum()))
            union = max(1, int((region.mask | tamper_mask).sum()))
            tamper_ratio = tamper_area / region_area
            region_iou = tamper_area / union
            if confidence_map is None:
                mask_confidence = 1.0
            else:
                mask_confidence = float(np.mean(confidence_map[region.mask])) if region.mask.any() else 0.0
            score = (
                self.config.tamper_ratio_weight * tamper_ratio
                + self.config.region_iou_weight * region_iou
                + self.config.confidence_weight * mask_confidence * min(1.0, tamper_ratio * 2.0)
            )
            if score >= self.config.select_threshold:
                candidates.append(TamperRegion(region, tamper_area, tamper_ratio, region_iou, mask_confidence, score))
        candidates.sort(key=lambda item: item.object_tamper_score, reverse=True)
        return candidates

