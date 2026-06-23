from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .schema import ObjectRegion, SeverityResult


SEVERITY_COLORS = {
    "low": (64, 180, 75),
    "medium": (255, 193, 7),
    "high": (245, 124, 0),
    "critical": (220, 53, 69),
}


def save_overlay(
    image: Image.Image,
    edit_mask: np.ndarray,
    regions: list[ObjectRegion],
    severities: list[SeverityResult],
    output_path: str | Path,
) -> None:
    base = np.asarray(image).astype(np.float32)
    overlay = base.copy()
    overlay[edit_mask] = 0.55 * overlay[edit_mask] + 0.45 * np.asarray([255, 0, 0], dtype=np.float32)
    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)

    for region, severity in zip(regions, severities):
        color = SEVERITY_COLORS.get(severity.level, (255, 255, 255))
        _draw_mask_contour(draw, region.object_mask, color)
        x0, y0, x1, y1 = region.bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = f"#{region.region_id} {severity.level} {severity.score:.1f}"
        draw.text((x0 + 2, max(0, y0 - 14)), label, fill=color)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def save_object_index_mask(regions: list[ObjectRegion], shape: tuple[int, int], output_path: str | Path) -> None:
    canvas = np.zeros(shape, dtype=np.uint16)
    for region in regions:
        canvas[region.object_mask] = region.region_id
    if canvas.max() == 0:
        rgb = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    else:
        normalized = (canvas.astype(np.float32) / float(canvas.max()) * 255).astype(np.uint8)
        rgb = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        rgb[canvas == 0] = 0
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)


def _draw_mask_contour(draw: ImageDraw.ImageDraw, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [(int(p[0][0]), int(p[0][1])) for p in contour]
        if len(points) > 1:
            draw.line(points + [points[0]], fill=color, width=2)
