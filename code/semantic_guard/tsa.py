from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .schema import ObjectRegion, bbox_from_mask


@dataclass
class TSAConfig:
    backend: str = "auto"
    sam_checkpoint: str | None = None
    sam_model_type: str = "vit_b"
    device: str = "auto"
    min_area: int = 32
    context_dilation: int = 5
    sam_min_iou: float = 0.05
    sam_component_coverage: float = 0.35
    sam_object_overlap: float = 0.03
    merge_iou: float = 0.75
    max_sam_masks_per_component: int = 4


class TamperedSemanticAggregator:
    """Aggregate EditGuard pixel masks into object-level tamper regions."""

    def __init__(self, config: TSAConfig):
        self.config = config
        self._sam_generator = None
        self._sam_error: str | None = None

    def extract(self, image: Image.Image, edit_mask: np.ndarray) -> list[ObjectRegion]:
        backend = self.config.backend.lower()
        if backend not in {"auto", "sam", "components"}:
            raise ValueError(f"Unsupported TSA backend: {self.config.backend}")

        if backend in {"auto", "sam"}:
            regions = self._extract_with_sam(image, edit_mask)
            if regions:
                return self._renumber(self._merge_regions(regions))
            if backend == "sam" and self._sam_error:
                raise RuntimeError(self._sam_error)

        return self._renumber(self._components_to_regions(edit_mask, "components"))

    def _extract_with_sam(self, image: Image.Image, edit_mask: np.ndarray) -> list[ObjectRegion]:
        if not self.config.sam_checkpoint:
            self._sam_error = "SAM checkpoint was not provided; falling back to connected components."
            return []
        try:
            generator = self._load_sam()
            sam_masks: list[dict[str, Any]] = generator.generate(np.asarray(image))
        except Exception as exc:  # optional dependency and checkpoint path are user supplied
            self._sam_error = f"SAM backend failed: {exc}"
            return []

        components = self._component_masks(edit_mask)
        if not components:
            return []

        regions: list[ObjectRegion] = []
        for component in components:
            comp_area = int(component.sum())
            candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
            for item in sam_masks:
                seg = np.asarray(item["segmentation"], dtype=bool)
                if seg.shape != edit_mask.shape:
                    seg = cv2.resize(
                        seg.astype(np.uint8),
                        (edit_mask.shape[1], edit_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                inter = int((seg & component).sum())
                if inter == 0:
                    continue
                seg_area = max(1, int(seg.sum()))
                union = max(1, int((seg | component).sum()))
                comp_cover = inter / max(1, comp_area)
                object_overlap = inter / seg_area
                iou = inter / union
                if (
                    iou >= self.config.sam_min_iou
                    or comp_cover >= self.config.sam_component_coverage
                    or object_overlap >= self.config.sam_object_overlap
                ):
                    pred_iou = float(item.get("predicted_iou", 0.0))
                    stability = float(item.get("stability_score", 0.0))
                    score = 0.45 * comp_cover + 0.25 * iou + 0.20 * pred_iou + 0.10 * stability
                    candidates.append((score, seg, item))

            if not candidates:
                regions.extend(self._components_to_regions(component, "components_fallback"))
                continue

            candidates.sort(key=lambda row: row[0], reverse=True)
            object_mask = np.zeros_like(edit_mask, dtype=bool)
            covered_component = np.zeros_like(edit_mask, dtype=bool)
            scores: list[float] = []
            for score, seg, _ in candidates[: self.config.max_sam_masks_per_component]:
                new_cover = int(((seg & component) & ~covered_component).sum())
                if object_mask.any() and new_cover / max(1, comp_area) < 0.03:
                    continue
                object_mask |= seg
                covered_component |= seg & component
                scores.append(float(score))
                if covered_component.sum() / max(1, comp_area) >= 0.96:
                    break

            object_mask |= component
            regions.append(
                ObjectRegion(
                    region_id=len(regions) + 1,
                    object_mask=object_mask,
                    edit_mask=edit_mask,
                    bbox=bbox_from_mask(object_mask),
                    source="sam_aggregation",
                    confidence=float(np.mean(scores)) if scores else 0.5,
                    notes=["SAM masks aggregated around EditGuard residual component."],
                )
            )

        return regions

    def _load_sam(self):
        if self._sam_generator is not None:
            return self._sam_generator

        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[self.config.sam_model_type](checkpoint=self.config.sam_checkpoint)
        sam.to(device=device)
        self._sam_generator = SamAutomaticMaskGenerator(
            sam,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.88,
            min_mask_region_area=self.config.min_area,
        )
        return self._sam_generator

    def _components_to_regions(self, mask: np.ndarray, source: str) -> list[ObjectRegion]:
        regions: list[ObjectRegion] = []
        for component in self._component_masks(mask):
            object_mask = component.copy()
            if self.config.context_dilation > 0:
                kernel_size = self.config.context_dilation * 2 + 1
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                object_mask = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
            regions.append(
                ObjectRegion(
                    region_id=len(regions) + 1,
                    object_mask=object_mask,
                    edit_mask=mask,
                    bbox=bbox_from_mask(object_mask),
                    source=source,
                    confidence=0.45 if source.startswith("components") else 0.6,
                    notes=["Connected-component fallback; not a true semantic segment."]
                    if source.startswith("components")
                    else [],
                )
            )
        return regions

    def _component_masks(self, mask: np.ndarray) -> list[np.ndarray]:
        mask_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        components: list[np.ndarray] = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < self.config.min_area:
                continue
            components.append(labels == label_id)
        return components

    def _merge_regions(self, regions: list[ObjectRegion]) -> list[ObjectRegion]:
        pending = regions[:]
        merged: list[ObjectRegion] = []
        while pending:
            base = pending.pop(0)
            changed = True
            while changed:
                changed = False
                kept: list[ObjectRegion] = []
                for other in pending:
                    inter = int((base.object_mask & other.object_mask).sum())
                    union = max(1, int((base.object_mask | other.object_mask).sum()))
                    if inter / union >= self.config.merge_iou:
                        base.object_mask = base.object_mask | other.object_mask
                        base.edit_mask = base.edit_mask | other.edit_mask
                        base.bbox = bbox_from_mask(base.object_mask)
                        base.confidence = max(base.confidence, other.confidence)
                        base.notes = list(dict.fromkeys(base.notes + other.notes + ["Merged overlapping TSA regions."]))
                        changed = True
                    else:
                        kept.append(other)
                pending = kept
            merged.append(base)
        return merged

    @staticmethod
    def _renumber(regions: list[ObjectRegion]) -> list[ObjectRegion]:
        for idx, region in enumerate(regions, start=1):
            region.region_id = idx
        return regions
