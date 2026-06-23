from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_v0_pipeline import run_pipeline


def create_toy_example(root: Path) -> Path:
    original_dir = root / "data" / "original"
    tampered_dir = root / "data" / "tampered"
    pred_dir = root / "data" / "masks" / "editguard_pred"
    gt_dir = root / "data" / "masks" / "gt"
    for directory in [original_dir, tampered_dir, pred_dir, gt_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    width, height = 320, 220
    original = Image.new("RGB", (width, height), (230, 235, 240))
    draw = ImageDraw.Draw(original)
    draw.rectangle([0, 150, width, height], fill=(125, 170, 120))
    draw.ellipse([55, 45, 135, 135], fill=(240, 205, 90), outline=(80, 60, 30), width=3)
    draw.rectangle([190, 70, 265, 145], fill=(80, 140, 220), outline=(20, 60, 120), width=3)
    draw.text((200, 95), "LOGO", fill=(255, 255, 255))

    tampered = original.copy()
    tamper_draw = ImageDraw.Draw(tampered)
    tamper_draw.rectangle([190, 70, 265, 145], fill=(220, 80, 90), outline=(120, 20, 30), width=3)
    tamper_draw.text((198, 95), "FAKE", fill=(255, 255, 255))

    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([190, 70, 265, 145], fill=255)

    original.save(original_dir / "demo.png")
    tampered.save(tampered_dir / "demo.png")
    mask.save(pred_dir / "demo.png")
    mask.save(gt_dir / "demo.png")

    metadata_path = root / "data" / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "original_path",
                "tampered_path",
                "editguard_mask_path",
                "gt_mask_path",
                "tamper_type",
                "gt_object_label",
                "gt_severity",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "demo",
                "original_path": "data/original/demo.png",
                "tampered_path": "data/tampered/demo.png",
                "editguard_mask_path": "data/masks/editguard_pred/demo.png",
                "gt_mask_path": "data/masks/gt/demo.png",
                "tamper_type": "logo_replace",
                "gt_object_label": "logo",
                "gt_severity": "critical",
            }
        )
    return metadata_path


def main() -> None:
    metadata_path = create_toy_example(PROJECT_ROOT)
    rows = run_pipeline(
        metadata_path=metadata_path,
        config_path=PROJECT_ROOT / "configs" / "default.yaml",
        output_dir=PROJECT_ROOT / "outputs",
        project_root=PROJECT_ROOT,
    )
    print(f"Demo complete. Rows: {len(rows)}")
    print(f"Results: {PROJECT_ROOT / 'outputs' / 'results.csv'}")
    print(f"Report: {PROJECT_ROOT / 'outputs' / 'reports' / 'demo.json'}")
    print(f"Visualization: {PROJECT_ROOT / 'outputs' / 'visualizations' / 'demo_vis.png'}")


if __name__ == "__main__":
    main()

