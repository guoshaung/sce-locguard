from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import Region, bbox_from_mask, load_mask


@dataclass
class RegionProposalConfig:
    mode: str = "connected_components"
    min_area: int = 32
    grid_size: int = 4
    sam_checkpoint: str | None = None
    sam_model_type: str = "vit_b"
    device: str = "cpu"


class RegionProposal:
    """Generate candidate semantic regions for TSA v0."""

    def __init__(self, config: RegionProposalConfig):
        self.config = config

    def propose(
        self,
        image: np.ndarray,
        editguard_mask: np.ndarray,
        gt_mask_path: str | Path | None = None,
    ) -> list[Region]:
        mode = self.config.mode.lower()
        if mode == "gt_mask":
            regions = self._from_gt_mask(gt_mask_path, editguard_mask.shape)
        elif mode == "grid":
            regions = self._grid_regions(editguard_mask.shape)
        elif mode == "sam":
            regions = self._sam_regions(image, editguard_mask)
            if not regions:
                regions = self._connected_components(editguard_mask)
        elif mode == "connected_components":
            regions = self._connected_components(editguard_mask)
        else:
            raise ValueError(f"Unsupported region proposal mode: {self.config.mode}")

        if not regions:
            regions = self._whole_image_region(editguard_mask.shape)
        return regions

    def _from_gt_mask(self, gt_mask_path: str | Path | None, shape: tuple[int, int]) -> list[Region]:
        if gt_mask_path is None:
            return []
        mask = load_mask(gt_mask_path, size=(shape[1], shape[0]))
        return self._connected_components(mask, source_label="gt")

    def _grid_regions(self, shape: tuple[int, int]) -> list[Region]:
        height, width = shape
        regions: list[Region] = []
        grid = max(1, self.config.grid_size)
        cell_h = int(np.ceil(height / grid))
        cell_w = int(np.ceil(width / grid))
        region_id = 1
        for gy in range(grid):
            for gx in range(grid):
                y1 = gy * cell_h
                x1 = gx * cell_w
                y2 = min(height, y1 + cell_h)
                x2 = min(width, x1 + cell_w)
                mask = np.zeros(shape, dtype=bool)
                mask[y1:y2, x1:x2] = True
                area = int(mask.sum())
                if area >= self.config.min_area:
                    regions.append(Region(region_id, mask, (x1, y1, x2, y2), area, label="grid", confidence=0.5))
                    region_id += 1
        return regions

    def _connected_components(self, mask: np.ndarray, source_label: str = "component") -> list[Region]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        regions: list[Region] = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < self.config.min_area:
                continue
            region_mask = labels == label_id
            regions.append(
                Region(
                    region_id=len(regions) + 1,
                    mask=region_mask,
                    bbox=bbox_from_mask(region_mask),
                    area=area,
                    label=source_label,
                    confidence=0.75,
                )
            )
        return regions

    def _sam_regions(self, image: np.ndarray, editguard_mask: np.ndarray) -> list[Region]:
        if not self.config.sam_checkpoint:
            return []
        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[self.config.sam_model_type](checkpoint=self.config.sam_checkpoint)
            sam.to(device=device)
            generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=self.config.min_area)
            sam_masks = generator.generate(image)
        except Exception:
            return []

        regions: list[Region] = []
        for item in sam_masks:
            mask = np.asarray(item["segmentation"], dtype=bool)
            overlap = int((mask & editguard_mask).sum())
            if overlap == 0:
                continue
            area = int(mask.sum())
            if area < self.config.min_area:
                continue
            regions.append(
                Region(
                    region_id=len(regions) + 1,
                    mask=mask,
                    bbox=bbox_from_mask(mask),
                    area=area,
                    label="sam",
                    confidence=float(item.get("predicted_iou", 0.8)),
                )
            )
        return regions

    @staticmethod
    def _whole_image_region(shape: tuple[int, int]) -> list[Region]:
        mask = np.ones(shape, dtype=bool)
        height, width = shape
        return [Region(1, mask, (0, 0, width, height), int(mask.sum()), label="whole_image", confidence=0.25)]

