from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

FRAGMENT_FIELDS = [
    "image_id",
    "num_pred_regions",
    "num_gt_regions",
    "fragmentation_ratio",
    "small_pred_region_count",
    "small_pred_region_ratio",
    "tiny_pred_region_count",
    "tiny_pred_region_ratio",
    "largest_pred_region_area_ratio",
    "total_pred_area_ratio",
    "semantic_compactness",
    "over_fragmentation_score",
    "iou",
    "dice",
    "precision",
    "recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EditGuard pred-mask semantic fragmentation from Stage 2/3 outputs.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--stage1_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--stage2_csv", default="dfg_locguard/outputs/stage2_pred_eval/per_sample_metrics.csv")
    parser.add_argument("--stage3_region_csv", default="dfg_locguard/outputs/stage3_semantic_eval_v1/per_region_semantic_metrics.csv")
    parser.add_argument("--stage3_sample_csv", default="dfg_locguard/outputs/stage3_semantic_eval_v1/per_sample_semantic_summary.csv")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage3_5_fragmentation_analysis")
    parser.add_argument("--small_area_pixels", type=int, default=100)
    parser.add_argument("--tiny_area_pixels", type=int, default=25)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def as_int(value: str, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_mask(path: Path, threshold: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > threshold


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (220, 220)) -> Image.Image:
    header_h = 28
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


def error_map(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[np.logical_and(pred, gt)] = [255, 255, 255]
    out[np.logical_and(pred, ~gt)] = [255, 80, 80]
    out[np.logical_and(~pred, gt)] = [80, 160, 255]
    return Image.fromarray(out, mode="RGB")


def component_overlay(original: Image.Image, pred: np.ndarray, max_boxes: int = 80) -> Image.Image:
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    overlay = base.copy()
    rng = np.random.default_rng(20260621)
    colors = rng.integers(60, 255, size=(max(labels_count, 1), 3), dtype=np.uint8)
    for label_id in range(1, labels_count):
        region = labels == label_id
        color = colors[label_id].astype(np.float32)
        overlay[region] = 0.35 * overlay[region] + 0.65 * color
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    ordered = sorted(range(1, labels_count), key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]), reverse=True)
    for label_id in ordered[:max_boxes]:
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        draw.rectangle([x, y, x + max(0, w - 1), y + max(0, h - 1)], outline=(255, 255, 0), width=1)
    return image


def save_fragmentation_visual(stage1_dir: Path, image_id: str, output_path: Path, threshold: int) -> None:
    sample_dir = stage1_dir / image_id
    original = Image.open(sample_dir / "original.png").convert("RGB")
    tampered = Image.open(sample_dir / "tampered.png").convert("RGB")
    gt = load_mask(sample_dir / "gt_mask.png", threshold)
    pred = load_mask(sample_dir / "editguard_pred_mask.png", threshold)
    if pred.shape != gt.shape:
        pred = np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize((gt.shape[1], gt.shape[0]), NEAREST)) > threshold
    panels = [
        title_panel(original, "original"),
        title_panel(tampered, "tampered"),
        title_panel(mask_image(gt), "gt_mask"),
        title_panel(mask_image(pred), "pred_mask"),
        title_panel(component_overlay(tampered, pred), "pred connected components"),
        title_panel(error_map(gt, pred), "error map"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_contact_sheet(vis_paths: list[Path], output_path: Path, max_samples: int) -> dict[str, Any]:
    selected = vis_paths[:max_samples]
    rows = []
    for path in selected:
        if path.exists():
            rows.append(Image.open(path).convert("RGB").resize((660, 124), BICUBIC))
    if not rows:
        return {"path": str(output_path.resolve()), "shown_samples": 0}
    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "shown_samples": len(rows)}


def build_metrics(
    region_rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
    pixel_rows: list[dict[str, str]],
    small_area_pixels: int,
    tiny_area_pixels: int,
) -> list[dict[str, Any]]:
    pred_regions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in region_rows:
        if row.get("status") == "ok" and row.get("mask_source") == "pred":
            pred_regions[row["image_id"]].append(row)

    sample_by_id = {row["image_id"]: row for row in sample_rows}
    pixel_by_id = {row["image_id"]: row for row in pixel_rows}
    metrics: list[dict[str, Any]] = []
    for image_id in sorted(sample_by_id):
        sample = sample_by_id[image_id]
        pixel = pixel_by_id.get(image_id, {})
        regions = pred_regions.get(image_id, [])
        num_pred_regions = as_int(sample.get("num_pred_regions", "0"))
        num_gt_regions = as_int(sample.get("num_gt_regions", "0"))
        areas = [as_int(row.get("area_pixels", "0")) for row in regions]
        area_ratios = [as_float(row.get("area_ratio", "0")) for row in regions]
        small_count = sum(area < small_area_pixels for area in areas)
        tiny_count = sum(area < tiny_area_pixels for area in areas)
        total_area = int(sum(areas))
        largest_area = int(max(areas, default=0))
        total_pred_area_ratio = float(sum(area_ratios))
        largest_pred_region_area_ratio = as_float(sample.get("largest_pred_region_area_ratio", "0"))
        semantic_compactness = float(largest_area / total_area) if total_area else 0.0
        fragmentation_ratio = float(num_pred_regions / max(num_gt_regions, 1))
        small_ratio = float(small_count / max(num_pred_regions, 1))
        tiny_ratio = float(tiny_count / max(num_pred_regions, 1))
        over_fragmentation_score = float(
            np.clip(
                0.45 * min(fragmentation_ratio / 10.0, 1.0)
                + 0.25 * small_ratio
                + 0.15 * tiny_ratio
                + 0.15 * (1.0 - semantic_compactness),
                0.0,
                1.0,
            )
        )
        metrics.append(
            {
                "image_id": image_id,
                "num_pred_regions": num_pred_regions,
                "num_gt_regions": num_gt_regions,
                "fragmentation_ratio": fragmentation_ratio,
                "small_pred_region_count": small_count,
                "small_pred_region_ratio": small_ratio,
                "tiny_pred_region_count": tiny_count,
                "tiny_pred_region_ratio": tiny_ratio,
                "largest_pred_region_area_ratio": largest_pred_region_area_ratio,
                "total_pred_area_ratio": total_pred_area_ratio,
                "semantic_compactness": semantic_compactness,
                "over_fragmentation_score": over_fragmentation_score,
                "iou": as_float(pixel.get("iou", "0")),
                "dice": as_float(pixel.get("dice", "0")),
                "precision": as_float(pixel.get("precision", "0")),
                "recall": as_float(pixel.get("recall", "0")),
            }
        )
    return metrics


def summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in metrics]

    worst = sorted(metrics, key=lambda row: (row["over_fragmentation_score"], row["fragmentation_ratio"]), reverse=True)[:20]
    return {
        "total_samples": len(metrics),
        "mean_fragmentation_ratio": float(np.mean(values("fragmentation_ratio"))) if metrics else 0.0,
        "median_fragmentation_ratio": float(np.median(values("fragmentation_ratio"))) if metrics else 0.0,
        "mean_num_pred_regions": float(np.mean(values("num_pred_regions"))) if metrics else 0.0,
        "mean_num_gt_regions": float(np.mean(values("num_gt_regions"))) if metrics else 0.0,
        "mean_small_pred_region_ratio": float(np.mean(values("small_pred_region_ratio"))) if metrics else 0.0,
        "mean_tiny_pred_region_ratio": float(np.mean(values("tiny_pred_region_ratio"))) if metrics else 0.0,
        "mean_semantic_compactness": float(np.mean(values("semantic_compactness"))) if metrics else 0.0,
        "correlation_iou_vs_fragmentation_ratio": pearson(values("iou"), values("fragmentation_ratio")),
        "correlation_dice_vs_fragmentation_ratio": pearson(values("dice"), values("fragmentation_ratio")),
        "top_20_worst_fragmented_image_ids": [row["image_id"] for row in worst],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    visual_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    region_rows = read_csv(project_root / args.stage3_region_csv)
    sample_rows = read_csv(project_root / args.stage3_sample_csv)
    pixel_rows = read_csv(project_root / args.stage2_csv)
    metrics = build_metrics(region_rows, sample_rows, pixel_rows, args.small_area_pixels, args.tiny_area_pixels)

    write_csv(output_dir / "per_sample_fragmentation_metrics.csv", metrics, FRAGMENT_FIELDS)
    worst = sorted(metrics, key=lambda row: (row["over_fragmentation_score"], row["fragmentation_ratio"]), reverse=True)
    best = sorted(metrics, key=lambda row: (row["over_fragmentation_score"], -row["semantic_compactness"]))
    write_csv(output_dir / "worst_fragmented_samples.csv", worst[:100], FRAGMENT_FIELDS)
    write_csv(output_dir / "best_compact_samples.csv", best[:100], FRAGMENT_FIELDS)

    vis_paths = []
    stage1_dir = project_root / args.stage1_dir
    for row in worst[: args.contact_sheet_samples]:
        image_id = row["image_id"]
        vis_path = visual_dir / f"{image_id}_fragmentation_vis.png"
        save_fragmentation_visual(stage1_dir, image_id, vis_path, args.mask_threshold)
        vis_paths.append(vis_path)

    payload = summary(metrics)
    payload["small_area_pixels"] = args.small_area_pixels
    payload["tiny_area_pixels"] = args.tiny_area_pixels
    payload["contact_sheet"] = build_contact_sheet(vis_paths, output_dir / "fragmentation_contact_sheet.png", args.contact_sheet_samples)
    (output_dir / "summary_fragmentation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved per-sample metrics: {output_dir / 'per_sample_fragmentation_metrics.csv'}")
    print(f"Saved summary: {output_dir / 'summary_fragmentation.json'}")
    print(f"Saved worst samples: {output_dir / 'worst_fragmented_samples.csv'}")
    print(f"Saved best samples: {output_dir / 'best_compact_samples.csv'}")
    print(f"Saved contact sheet: {output_dir / 'fragmentation_contact_sheet.png'}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
