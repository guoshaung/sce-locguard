from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - optional dependency
    skimage_ssim = None


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

REGION_FIELDS = [
    "image_id",
    "mask_source",
    "region_id",
    "connected_component_count",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "area_pixels",
    "area_ratio",
    "centroid_x",
    "centroid_y",
    "overlap_iou",
    "overlap_with_gt_iou",
    "overlap_with_pred_iou",
    "mean_abs_rgb_diff",
    "mean_abs_gray_diff",
    "max_abs_gray_diff",
    "changed_pixel_ratio",
    "ssim_region",
    "scc_change_score",
    "scc_mode",
    "scc_change_level",
    "severity_score",
    "severity_level",
    "status",
    "error_message",
]

SAMPLE_FIELDS = [
    "image_id",
    "num_pred_regions",
    "num_gt_regions",
    "largest_pred_region_area_ratio",
    "largest_gt_region_area_ratio",
    "mean_region_change_score",
    "max_region_change_score",
    "overall_scc_change_level",
    "overall_severity_level",
    "status",
    "error_message",
]

CHANGE_LEVEL_ORDER = {"none": 0, "weak": 1, "medium": 2, "strong": 3}
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight Stage 3 TSA/SCC/SSE semantic evaluation v1.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--input_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage3_semantic_eval_v1")
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--changed_pixel_threshold", type=float, default=15.0)
    parser.add_argument("--none_change_score", type=float, default=0.02)
    parser.add_argument("--weak_change_score", type=float, default=0.15)
    parser.add_argument("--medium_change_score", type=float, default=0.40)
    parser.add_argument("--medium_area_ratio", type=float, default=0.02)
    parser.add_argument("--large_area_ratio", type=float, default=0.08)
    parser.add_argument("--critical_area_ratio", type=float, default=0.20)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--max_visual_regions", type=int, default=50)
    return parser.parse_args()


def sample_dirs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit())


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, threshold: int, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, NEAREST)
    return np.asarray(mask, dtype=np.uint8) > threshold


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(inter / union) if union else 0.0


def connected_regions(mask: np.ndarray, mask_source: str, gt_mask: np.ndarray, pred_mask: np.ndarray) -> list[dict[str, Any]]:
    mask_u8 = mask.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    h, w = mask.shape
    regions: list[dict[str, Any]] = []
    component_count = max(0, count - 1)
    for label_id in range(1, count):
        x, y, bw, bh, area = stats[label_id].tolist()
        region_mask = labels == label_id
        overlap_with_gt_iou = mask_iou(region_mask, gt_mask)
        overlap_with_pred_iou = mask_iou(region_mask, pred_mask)
        regions.append(
            {
                "mask_source": mask_source,
                "region_id": label_id,
                "connected_component_count": component_count,
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "area_pixels": int(area),
                "area_ratio": float(area / max(1, h * w)),
                "centroid": [float(centroids[label_id][0]), float(centroids[label_id][1])],
                "mask": region_mask,
                "overlap_with_gt_iou": overlap_with_gt_iou,
                "overlap_with_pred_iou": overlap_with_pred_iou,
                "overlap_iou": overlap_with_gt_iou if mask_source == "pred" else overlap_with_pred_iou,
            }
        )
    return regions


def rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)


def region_ssim(gray_a: np.ndarray, gray_b: np.ndarray, bbox: list[int]) -> float | None:
    if skimage_ssim is None:
        return None
    x1, y1, x2, y2 = bbox
    crop_a = gray_a[y1:y2, x1:x2]
    crop_b = gray_b[y1:y2, x1:x2]
    if crop_a.size == 0 or min(crop_a.shape[:2]) < 7:
        return None
    return float(skimage_ssim(crop_a, crop_b, data_range=255))


def change_level(change_score: float, args: argparse.Namespace) -> str:
    if change_score < args.none_change_score:
        return "none"
    if change_score < args.weak_change_score:
        return "weak"
    if change_score < args.medium_change_score:
        return "medium"
    return "strong"


def compute_scc(
    original_arr: np.ndarray,
    tampered_arr: np.ndarray,
    original_gray: np.ndarray,
    tampered_gray: np.ndarray,
    region: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    region_mask = region["mask"]
    if not region_mask.any():
        return {
            "mean_abs_rgb_diff": 0.0,
            "mean_abs_gray_diff": 0.0,
            "max_abs_gray_diff": 0.0,
            "changed_pixel_ratio": 0.0,
            "ssim_region": None,
            "scc_change_score": 0.0,
            "scc_mode": "reference_based",
            "scc_change_level": "none",
        }

    rgb_diff = np.abs(original_arr.astype(np.float32) - tampered_arr.astype(np.float32))
    gray_diff = np.abs(original_gray - tampered_gray)
    region_rgb = rgb_diff[region_mask]
    region_gray = gray_diff[region_mask]
    mean_abs_rgb_diff = float(region_rgb.mean()) if region_rgb.size else 0.0
    mean_abs_gray_diff = float(region_gray.mean()) if region_gray.size else 0.0
    max_abs_gray_diff = float(region_gray.max()) if region_gray.size else 0.0
    changed_pixel_ratio = float((region_gray > args.changed_pixel_threshold).mean()) if region_gray.size else 0.0
    gray_score = min(mean_abs_gray_diff / 80.0, 1.0)
    change_score = float(np.clip(0.5 * gray_score + 0.5 * changed_pixel_ratio, 0.0, 1.0))
    return {
        "mean_abs_rgb_diff": mean_abs_rgb_diff,
        "mean_abs_gray_diff": mean_abs_gray_diff,
        "max_abs_gray_diff": max_abs_gray_diff,
        "changed_pixel_ratio": changed_pixel_ratio,
        "ssim_region": region_ssim(original_gray, tampered_gray, region["bbox"]),
        "scc_change_score": change_score,
        "scc_mode": "reference_based",
        "scc_change_level": change_level(change_score, args),
    }


def severity(region: dict[str, Any], scc: dict[str, Any], image_size: tuple[int, int], args: argparse.Namespace) -> dict[str, Any]:
    area_ratio = float(region["area_ratio"])
    change = str(scc["scc_change_level"])
    change_weight = {"none": 0.0, "weak": 0.25, "medium": 0.55, "strong": 0.90}[change]
    area_score = min(area_ratio / max(args.critical_area_ratio, 1e-8), 1.0)
    w, h = image_size
    cx, cy = region["centroid"]
    center_dist = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
    max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    center_score = float(1.0 - min(center_dist / max_dist, 1.0))
    score = float(np.clip(100.0 * (0.45 * change_weight + 0.40 * area_score + 0.15 * center_score), 0.0, 100.0))

    if area_ratio >= args.critical_area_ratio and change == "strong":
        level = "critical"
    elif area_ratio >= args.large_area_ratio or change == "strong":
        level = "high"
    elif area_ratio >= args.medium_area_ratio or change == "medium":
        level = "medium"
    else:
        level = "low"
    return {"severity_score": score, "severity_level": level}


def row_from_region(
    image_id: str,
    region: dict[str, Any],
    scc: dict[str, Any],
    sev: dict[str, Any],
) -> dict[str, Any]:
    x1, y1, x2, y2 = region["bbox"]
    cx, cy = region["centroid"]
    return {
        "image_id": image_id,
        "mask_source": region["mask_source"],
        "region_id": region["region_id"],
        "connected_component_count": region["connected_component_count"],
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "area_pixels": region["area_pixels"],
        "area_ratio": region["area_ratio"],
        "centroid_x": cx,
        "centroid_y": cy,
        "overlap_iou": region["overlap_iou"],
        "overlap_with_gt_iou": region["overlap_with_gt_iou"],
        "overlap_with_pred_iou": region["overlap_with_pred_iou"],
        "mean_abs_rgb_diff": scc["mean_abs_rgb_diff"],
        "mean_abs_gray_diff": scc["mean_abs_gray_diff"],
        "max_abs_gray_diff": scc["max_abs_gray_diff"],
        "changed_pixel_ratio": scc["changed_pixel_ratio"],
        "ssim_region": "" if scc["ssim_region"] is None else scc["ssim_region"],
        "scc_change_score": scc["scc_change_score"],
        "scc_mode": scc["scc_mode"],
        "scc_change_level": scc["scc_change_level"],
        "severity_score": sev["severity_score"],
        "severity_level": sev["severity_level"],
        "status": "ok",
        "error_message": "",
    }


def strongest_level(rows: list[dict[str, Any]], field: str, order: dict[str, int], default: str) -> str:
    if not rows:
        return default
    return max((str(row[field]) for row in rows), key=lambda level: order.get(level, -1))


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


def heatmap_image(original_arr: np.ndarray, tampered_arr: np.ndarray) -> Image.Image:
    diff = np.abs(original_arr.astype(np.float32) - tampered_arr.astype(np.float32)).max(axis=2)
    norm = diff / max(float(diff.max()), 1.0)
    heat = np.zeros((*diff.shape, 3), dtype=np.uint8)
    heat[:, :, 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
    heat[:, :, 1] = np.clip(norm * 80, 0, 80).astype(np.uint8)
    return Image.fromarray(heat, mode="RGB")


def pred_region_overlay(
    tampered: Image.Image,
    pred_mask: np.ndarray,
    pred_rows: list[dict[str, Any]],
    max_visual_regions: int,
) -> Image.Image:
    base = np.asarray(tampered.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    overlay[pred_mask] = 0.45 * overlay[pred_mask] + 0.55 * np.asarray([255, 60, 60], dtype=np.float32)
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    rows = sorted(pred_rows, key=lambda row: float(row["area_pixels"]), reverse=True)[:max_visual_regions]
    for row in rows:
        x1, y1, x2, y2 = int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])
        draw.rectangle([x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)], outline=(255, 255, 0), width=2)
        draw.text((x1 + 2, y1 + 2), f"{row['region_id']}:{row['severity_level']}", fill=(255, 255, 0))
    return image


def save_visualization(
    sample_dir: Path,
    output_path: Path,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_rows: list[dict[str, Any]],
    sample_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    original = load_rgb(sample_dir / "original.png")
    tampered = load_rgb(sample_dir / "tampered.png").resize(original.size, BICUBIC)
    original_arr = np.asarray(original, dtype=np.uint8)
    tampered_arr = np.asarray(tampered, dtype=np.uint8)
    severity_title = f"heatmap severity={sample_summary['overall_severity_level']}"
    panels = [
        title_panel(original, "original"),
        title_panel(tampered, "tampered"),
        title_panel(mask_image(pred_mask), "pred_mask"),
        title_panel(mask_image(gt_mask), "gt_mask"),
        title_panel(pred_region_overlay(tampered, pred_mask, pred_rows, args.max_visual_regions), "pred regions"),
        title_panel(heatmap_image(original_arr, tampered_arr), severity_title),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def evaluate_sample(sample_dir: Path, output_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    image_id = sample_dir.name
    required = ["original.png", "tampered.png", "gt_mask.png", "editguard_pred_mask.png"]
    missing = [name for name in required if not (sample_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    original = load_rgb(sample_dir / "original.png")
    tampered = load_rgb(sample_dir / "tampered.png").resize(original.size, BICUBIC)
    original_arr = np.asarray(original, dtype=np.uint8)
    tampered_arr = np.asarray(tampered, dtype=np.uint8)
    original_gray = rgb_to_gray(original_arr)
    tampered_gray = rgb_to_gray(tampered_arr)
    pred_mask = load_mask(sample_dir / "editguard_pred_mask.png", args.mask_threshold, size=original.size)
    gt_mask = load_mask(sample_dir / "gt_mask.png", args.mask_threshold, size=original.size)

    pred_regions = connected_regions(pred_mask, "pred", gt_mask, pred_mask)
    gt_regions = connected_regions(gt_mask, "gt", gt_mask, pred_mask)

    region_rows: list[dict[str, Any]] = []
    report_regions: list[dict[str, Any]] = []
    for region in pred_regions + gt_regions:
        scc = compute_scc(original_arr, tampered_arr, original_gray, tampered_gray, region, args)
        sev = severity(region, scc, original.size, args)
        row = row_from_region(image_id, region, scc, sev)
        region_rows.append(row)
        report_regions.append({key: value for key, value in row.items() if key not in {"image_id", "status", "error_message"}})

    pred_rows = [row for row in region_rows if row["mask_source"] == "pred"]
    gt_rows = [row for row in region_rows if row["mask_source"] == "gt"]
    pred_change_scores = [float(row["scc_change_score"]) for row in pred_rows]
    sample_summary = {
        "image_id": image_id,
        "num_pred_regions": len(pred_rows),
        "num_gt_regions": len(gt_rows),
        "largest_pred_region_area_ratio": max([float(row["area_ratio"]) for row in pred_rows], default=0.0),
        "largest_gt_region_area_ratio": max([float(row["area_ratio"]) for row in gt_rows], default=0.0),
        "mean_region_change_score": float(np.mean(pred_change_scores)) if pred_change_scores else 0.0,
        "max_region_change_score": max(pred_change_scores, default=0.0),
        "overall_scc_change_level": strongest_level(pred_rows, "scc_change_level", CHANGE_LEVEL_ORDER, "none"),
        "overall_severity_level": strongest_level(pred_rows, "severity_level", SEVERITY_ORDER, "low"),
        "status": "ok",
        "error_message": "",
    }
    report = {
        "image_id": image_id,
        "paths": {
            "original": str((sample_dir / "original.png").resolve()),
            "tampered": str((sample_dir / "tampered.png").resolve()),
            "pred_mask": str((sample_dir / "editguard_pred_mask.png").resolve()),
            "gt_mask": str((sample_dir / "gt_mask.png").resolve()),
        },
        "config": {
            "mask_threshold": args.mask_threshold,
            "changed_pixel_threshold": args.changed_pixel_threshold,
            "scc_mode": "reference_based",
            "semantic_backbone": "lightweight_pixel_statistics",
            "uses_sam_clip_dino_stable_diffusion": False,
        },
        "sample_summary": sample_summary,
        "regions": report_regions,
    }

    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / f"{image_id}_semantic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_visualization(
        sample_dir,
        output_dir / "visualizations" / f"{image_id}_semantic_vis.png",
        pred_mask,
        gt_mask,
        pred_rows,
        sample_summary,
        args,
    )
    return region_rows, sample_summary, report


def failed_region_row(image_id: str, message: str) -> dict[str, Any]:
    return {field: "" for field in REGION_FIELDS} | {
        "image_id": image_id,
        "status": "failed",
        "error_message": message,
    }


def failed_sample_row(image_id: str, message: str) -> dict[str, Any]:
    return {field: "" for field in SAMPLE_FIELDS} | {
        "image_id": image_id,
        "status": "failed",
        "error_message": message,
    }


def summarize(sample_rows: list[dict[str, Any]], region_rows: list[dict[str, Any]], failed: list[dict[str, str]]) -> dict[str, Any]:
    ok_samples = [row for row in sample_rows if row["status"] == "ok"]
    ok_regions = [row for row in region_rows if row["status"] == "ok"]
    pred_regions = [row for row in ok_regions if row["mask_source"] == "pred"]
    gt_regions = [row for row in ok_regions if row["mask_source"] == "gt"]
    severity_counts = Counter(str(row["severity_level"]) for row in pred_regions)
    scc_counts = Counter(str(row["scc_change_level"]) for row in pred_regions)
    return {
        "total_samples": len(sample_rows),
        "evaluated_samples": len(ok_samples),
        "failed_samples": len(failed),
        "total_pred_regions": len(pred_regions),
        "total_gt_regions": len(gt_regions),
        "mean_pred_regions_per_sample": float(np.mean([int(row["num_pred_regions"]) for row in ok_samples])) if ok_samples else 0.0,
        "mean_gt_regions_per_sample": float(np.mean([int(row["num_gt_regions"]) for row in ok_samples])) if ok_samples else 0.0,
        "scc_reference_based_count": len(pred_regions),
        "severity_level_counts": dict(sorted(severity_counts.items())),
        "scc_change_level_counts": dict(sorted(scc_counts.items())),
    }


def build_contact_sheet(vis_paths: list[Path], output_path: Path, max_samples: int) -> dict[str, Any]:
    if len(vis_paths) > max_samples:
        indices = sorted({round(i * (len(vis_paths) - 1) / (max_samples - 1)) for i in range(max_samples)})
        selected = [vis_paths[index] for index in indices]
        sampling = "even"
    else:
        selected = vis_paths
        sampling = "all"
    rows = []
    for path in selected:
        if path.exists():
            rows.append(Image.open(path).convert("RGB").resize((660, 124), BICUBIC))
    if not rows:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}
    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(rows)}


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    for subdir in ["reports", "visualizations"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    all_region_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    dirs = sample_dirs(input_dir)
    for idx, sample_dir in enumerate(dirs, start=1):
        image_id = sample_dir.name
        try:
            region_rows, sample_summary, _ = evaluate_sample(sample_dir, output_dir, args)
            all_region_rows.extend(region_rows)
            sample_rows.append(sample_summary)
        except Exception as exc:
            message = str(exc)
            all_region_rows.append(failed_region_row(image_id, message))
            sample_rows.append(failed_sample_row(image_id, message))
            failed.append({"image_id": image_id, "failure_stage": "stage3_semantic_eval", "error_message": message})
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(dirs)} semantic eval done, failed={len(failed)}")

    region_csv = output_dir / "per_region_semantic_metrics.csv"
    with region_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGION_FIELDS)
        writer.writeheader()
        writer.writerows(all_region_rows)

    sample_csv = output_dir / "per_sample_semantic_summary.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(sample_rows)

    summary = summarize(sample_rows, all_region_rows, failed)
    vis_paths = [output_dir / "visualizations" / f"{row['image_id']}_semantic_vis.png" for row in sample_rows if row["status"] == "ok"]
    summary["contact_sheet"] = build_contact_sheet(
        vis_paths,
        output_dir / "stage3_semantic_contact_sheet.png",
        args.contact_sheet_samples,
    )
    (output_dir / "summary_semantic_eval.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved per-region CSV: {region_csv}")
    print(f"Saved per-sample CSV: {sample_csv}")
    print(f"Saved summary: {output_dir / 'summary_semantic_eval.json'}")
    print(f"Saved failed cases: {output_dir / 'failed_cases.json'}")
    print(f"Saved contact sheet: {output_dir / 'stage3_semantic_contact_sheet.png'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
