from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .common import SemanticResult, SeverityResult, TamperRegion


COLORS = {
    "low": (64, 180, 75),
    "medium": (255, 193, 7),
    "high": (245, 124, 0),
    "critical": (220, 53, 69),
}


def make_visualization(
    original: Image.Image,
    tampered: Image.Image,
    editguard_mask: np.ndarray,
    results: list[tuple[TamperRegion, SemanticResult, SeverityResult]],
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    panels = [
        _title_panel(original, "original"),
        _title_panel(tampered, "tampered"),
        _title_panel(Image.fromarray(editguard_mask.astype(np.uint8) * 255).convert("RGB"), "EditGuard mask"),
        _title_panel(_overlay(tampered, editguard_mask, results, alpha), "semantic evidence"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _title_panel(image: Image.Image, title: str) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + 24), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((5, 5), title, fill=(20, 20, 20))
    panel.paste(image.convert("RGB"), (0, 24))
    return panel


def _overlay(
    tampered: Image.Image,
    editguard_mask: np.ndarray,
    results: list[tuple[TamperRegion, SemanticResult, SeverityResult]],
    alpha: float,
) -> Image.Image:
    base = np.asarray(tampered).astype(np.float32)
    red = np.asarray([255, 0, 0], dtype=np.float32)
    base[editguard_mask] = (1 - alpha) * base[editguard_mask] + alpha * red
    out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for tamper_region, semantic, severity in results:
        color = COLORS.get(severity.severity_level, (255, 255, 255))
        _draw_contours(draw, tamper_region.region.mask, color)
        x1, y1, x2, y2 = tamper_region.region.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = (
            f"#{tamper_region.region.region_id} {tamper_region.region.label} "
            f"r={tamper_region.tamper_ratio:.2f} c={semantic.semantic_change_score:.2f} "
            f"{severity.severity_level}"
        )
        text_x = x1 + 2
        text_y = max(0, y1 - 16)
        try:
            text_box = draw.textbbox((text_x, text_y), label)
            if text_box[2] > out.width:
                text_x = max(0, out.width - (text_box[2] - text_box[0]) - 4)
        except Exception:
            pass
        draw.text((text_x, text_y), label, fill=color)
    return out


def _draw_contours(draw: ImageDraw.ImageDraw, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [(int(p[0][0]), int(p[0][1])) for p in contour]
        if len(points) > 1:
            draw.line(points + [points[0]], fill=color, width=2)
