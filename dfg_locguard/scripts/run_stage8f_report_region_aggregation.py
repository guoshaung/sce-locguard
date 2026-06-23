from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stage8e_report_quality_audit as s8e


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
NEAREST = getattr(Image, "Resampling", Image).NEAREST

REGION_FIELDS = [
    "strategy",
    "image_id",
    "aggregated_region_id",
    "source_region_ids",
    "num_source_regions",
    "bbox",
    "area_pixels",
    "area_ratio",
    "size_bucket",
    "center_score",
    "texture_score",
    "contrast_score",
    "edge_density",
    "contour_complexity",
    "severity",
    "severity_score",
    "change_type",
    "confidence",
    "stage8e_source_change_types",
    "copyright_valid",
    "capsule_valid",
    "auth_status",
    "payload_bits",
    "status",
]

SAMPLE_FIELDS = [
    "strategy",
    "image_id",
    "copyright_valid",
    "capsule_valid",
    "auth_status",
    "num_raw_regions",
    "num_filtered_regions",
    "num_removed_regions",
    "report_reduction_ratio",
    "pred_area_retained",
    "gt_tamper_area_coverage",
    "aggregated_iou",
    "aggregated_dice",
    "aggregated_precision",
    "aggregated_recall",
    "aggregated_mae",
    "unknown_rate_before",
    "unknown_rate_after_filter",
    "num_valid_region_reports",
    "num_no_report_image",
    "main_change_types",
    "max_severity",
    "overall_risk",
    "severity_distribution",
    "change_type_distribution",
    "report_summary",
]

STRATEGY_FIELDS = [
    "strategy",
    "total_images",
    "total_raw_regions",
    "total_filtered_regions",
    "report_reduction_ratio",
    "unknown_rate_before",
    "unknown_rate_after",
    "unknown_reduction_absolute",
    "unknown_reduction_relative",
    "mean_pred_area_retained",
    "mean_gt_tamper_area_coverage",
    "mean_aggregated_iou",
    "mean_aggregated_dice",
    "no_report_images",
    "valid_report_coverage_rate",
    "mean_reports_per_image",
    "severity_distribution_stability",
    "change_type_distribution_stability",
    "meets_success_criteria",
    "anti_cheating_flag",
    "recommended_by_tradeoff",
    "selection_score",
    "is_best",
]

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    area_threshold: float = 0.0
    top_k: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 8F report-level region aggregation and fragment filtering.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--stage8e_dir", default="dfg_locguard/outputs/stage8e_report_quality_audit")
    parser.add_argument("--stage1_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage8f_report_region_aggregation")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--merge_distance", type=float, default=48.0)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--unknown_gallery_samples", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def strategies() -> list[Strategy]:
    return [
        Strategy("F0_no_filter", "no_filter"),
        Strategy("F1_tiny_filter_ar0p0001", "tiny_filter", area_threshold=0.0001),
        Strategy("F1_tiny_filter_ar0p0005", "tiny_filter", area_threshold=0.0005),
        Strategy("F1_tiny_filter_ar0p001", "tiny_filter", area_threshold=0.001),
        Strategy("F2_low_confidence_tiny_filter", "low_confidence_tiny_filter"),
        Strategy("F3_merge_nearby_fragments", "merge_nearby_fragments"),
        Strategy("F4_main_region_top1", "main_region_report", top_k=1),
        Strategy("F4_main_region_top3", "main_region_report", top_k=3),
        Strategy("F4_main_region_top5", "main_region_report", top_k=5),
    ]


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "filtered_unknown_case_gallery": output_dir / "filtered_unknown_case_gallery",
        "visualizations": output_dir / "visualizations",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                out[field] = value
            writer.writerow(out)


def safe_float(value: Any, default: float = 0.0) -> float:
    return s8e.safe_float(value, default)


def safe_int(value: Any, default: int = 0) -> int:
    return s8e.safe_int(value, default)


def load_mask(path: Path, threshold: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((512, 512), NEAREST)
    return np.asarray(mask, dtype=np.uint8) > threshold


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)


def pixel_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())
    total = tp + fp + fn + tn
    return {
        "aggregated_iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0,
        "aggregated_dice": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "aggregated_precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "aggregated_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "aggregated_mae": float((fp + fn) / total) if total else 0.0,
    }


def parse_bbox(row: dict[str, Any]) -> list[int]:
    return [int(v) for v in s8e.parse_json_field(row.get("bbox"), [0, 0, 1, 1])]


def bbox_distance(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return float(math.sqrt(dx * dx + dy * dy))


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def bbox_center_distance(a: list[int], b: list[int]) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def region_confidence(row: dict[str, Any]) -> float:
    explicit = row.get("stage8e_confidence")
    if explicit not in ("", None):
        return safe_float(explicit)
    area_score = min(1.0, safe_float(row.get("area_ratio")) / 0.001)
    contrast = safe_float(row.get("contrast_score"))
    texture = safe_float(row.get("texture_score"))
    known_bonus = 0.25 if row.get("stage8e_change_type") != "unknown" else 0.0
    severity_bonus = 0.15 if row.get("severity") in {"high", "critical"} else 0.0
    return float(np.clip(0.25 + 0.25 * area_score + 0.20 * contrast + 0.15 * texture + known_bonus + severity_bonus, 0.0, 1.0))


def union_find_merge(rows: list[dict[str, Any]], distance: float) -> list[list[dict[str, Any]]]:
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    merge_candidates = [idx for idx, row in enumerate(rows) if row.get("size_bucket") in {"tiny", "small"}]
    for i, idx_a in enumerate(merge_candidates):
        bbox_a = parse_bbox(rows[idx_a])
        for idx_b in merge_candidates[i + 1 :]:
            bbox_b = parse_bbox(rows[idx_b])
            if (
                bbox_iou(bbox_a, bbox_b) > 0.0
                or bbox_distance(bbox_a, bbox_b) <= distance
                or bbox_center_distance(bbox_a, bbox_b) <= distance * 1.5
            ):
                union(idx_a, idx_b)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[find(idx)].append(row)
    return list(groups.values())


def select_groups(rows: list[dict[str, Any]], strategy: Strategy, merge_distance: float) -> list[list[dict[str, Any]]]:
    if strategy.family == "no_filter":
        return [[row] for row in rows]
    if strategy.family == "tiny_filter":
        kept = [row for row in rows if safe_float(row.get("area_ratio")) >= strategy.area_threshold]
        return [[row] for row in kept]
    if strategy.family == "low_confidence_tiny_filter":
        kept = [
            row
            for row in rows
            if not (row.get("size_bucket") == "tiny" and region_confidence(row) < 0.5)
        ]
        return [[row] for row in kept]
    if strategy.family == "merge_nearby_fragments":
        return union_find_merge(rows, merge_distance)
    if strategy.family == "main_region_report":
        kept = sorted(
            rows,
            key=lambda row: (safe_int(row.get("area_pixels")), region_confidence(row)),
            reverse=True,
        )[: strategy.top_k]
        return [[row] for row in kept]
    raise ValueError(f"Unsupported strategy: {strategy.name}")


def mask_for_group(labels: np.ndarray, group: list[dict[str, Any]]) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=bool)
    for row in group:
        region_id = safe_int(row.get("region_id"))
        out |= labels == region_id
    return out


def severity_from_score(score: float) -> str:
    if score < 0.15:
        return "low"
    if score < 0.40:
        return "medium"
    if score < 0.65:
        return "high"
    return "critical"


def aggregate_group(
    strategy: Strategy,
    image_id: str,
    agg_id: int,
    group: list[dict[str, Any]],
    group_mask: np.ndarray,
    tampered: np.ndarray,
) -> dict[str, Any]:
    ys, xs = np.where(group_mask)
    source_ids = [safe_int(row.get("region_id")) for row in group]
    first = group[0]
    if xs.size == 0:
        bbox = [0, 0, 1, 1]
        area = 0
        crop = np.zeros((1, 1, 3), dtype=np.uint8)
    else:
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bbox = [x1, y1, x2, y2]
        area = int(group_mask.sum())
        crop = tampered[y1:y2, x1:x2]
    image_area = float(group_mask.size)
    features = s8e.region_image_features(crop)
    center = s8e.center_score_from_bbox(bbox)
    area_ratio = float(area / image_area)
    area_score = float(np.clip(area_ratio / 0.20, 0.0, 1.0))
    severity_score = float(
        0.45 * area_score
        + 0.20 * center
        + 0.20 * features["texture_score"]
        + 0.15 * features["contrast_score"]
    )
    severity = severity_from_score(severity_score)
    representative = max(group, key=lambda row: (region_confidence(row), safe_int(row.get("area_pixels"))))
    classifier_row = dict(representative)
    classifier_row.update(
        {
            "bbox": bbox,
            "area_pixels": area,
            "area_ratio": area_ratio,
            "size_bucket": s8e.bucket_size(area),
            "center_score": center,
            "severity": severity,
            "severity_score": severity_score,
        }
    )
    _state, change_type, confidence, _reason = s8e.improved_change_type(classifier_row, features)
    source_change_types = [str(row.get("stage8e_change_type")) for row in group]
    if strategy.family != "merge_nearby_fragments" and len(group) == 1:
        change_type = str(first.get("stage8e_change_type"))
        severity = str(first.get("severity"))
        severity_score = safe_float(first.get("severity_score"))
        confidence = region_confidence(first)
    return {
        "strategy": strategy.name,
        "image_id": image_id,
        "aggregated_region_id": agg_id,
        "source_region_ids": source_ids,
        "num_source_regions": len(source_ids),
        "bbox": bbox,
        "area_pixels": area,
        "area_ratio": area_ratio,
        "size_bucket": s8e.bucket_size(area),
        "center_score": center,
        "texture_score": features["texture_score"],
        "contrast_score": features["contrast_score"],
        "edge_density": features["edge_density"],
        "contour_complexity": features["contour_complexity"],
        "severity": severity,
        "severity_score": severity_score,
        "change_type": change_type,
        "confidence": confidence,
        "stage8e_source_change_types": source_change_types,
        "copyright_valid": first.get("copyright_valid"),
        "capsule_valid": first.get("capsule_valid"),
        "auth_status": first.get("auth_status"),
        "payload_bits": first.get("payload_bits"),
        "status": "ok",
    }


def max_severity(labels: list[str]) -> str:
    if not labels:
        return "none"
    return max(labels, key=lambda label: SEVERITY_ORDER.get(label, -1))


def risk_from_reports(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "none"
    max_label = max_severity([str(row.get("severity")) for row in reports])
    unknown_rate = sum(1 for row in reports if row.get("change_type") == "unknown") / max(len(reports), 1)
    if max_label == "critical":
        return "critical"
    if max_label == "high":
        return "high"
    if unknown_rate > 0.5:
        return "uncertain"
    if max_label == "medium":
        return "medium"
    return "low"


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "")) for row in rows))


def total_variation(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a) | set(b)
    total_a = sum(a.values())
    total_b = sum(b.values())
    if total_a == 0 and total_b == 0:
        return 0.0
    return float(
        0.5
        * sum(abs(a.get(key, 0) / max(total_a, 1) - b.get(key, 0) / max(total_b, 1)) for key in keys)
    )


def process_strategy_for_image(
    strategy: Strategy,
    image_id: str,
    rows: list[dict[str, Any]],
    stage1_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    pred_path = stage1_dir / image_id / "editguard_pred_mask.png"
    gt_path = stage1_dir / image_id / "gt_mask.png"
    tampered_path = stage1_dir / image_id / "tampered.png"
    pred_mask = load_mask(pred_path, args.mask_threshold)
    gt_mask = load_mask(gt_path, args.mask_threshold)
    tampered = load_rgb(tampered_path)
    _num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
    groups = select_groups(rows, strategy, args.merge_distance)
    aggregated_mask = np.zeros_like(pred_mask, dtype=bool)
    region_reports: list[dict[str, Any]] = []
    for agg_id, group in enumerate(groups, start=1):
        group_mask = mask_for_group(labels, group)
        if int(group_mask.sum()) == 0:
            continue
        aggregated_mask |= group_mask
        region_reports.append(aggregate_group(strategy, image_id, agg_id, group, group_mask, tampered))
    raw_count = len(rows)
    filtered_count = len(region_reports)
    unknown_before = sum(1 for row in rows if row.get("stage8e_change_type") == "unknown") / max(raw_count, 1)
    unknown_after = sum(1 for row in region_reports if row.get("change_type") == "unknown") / max(filtered_count, 1)
    pix = pixel_metrics(gt_mask, aggregated_mask)
    pred_area_retained = float(aggregated_mask.sum() / max(int(pred_mask.sum()), 1))
    gt_coverage = float(np.logical_and(aggregated_mask, gt_mask).sum() / max(int(gt_mask.sum()), 1))
    change_dist = distribution(region_reports, "change_type")
    severity_dist = distribution(region_reports, "severity")
    main_change_types = [name for name, _count in Counter(row.get("change_type") for row in region_reports).most_common(3)]
    max_sev = max_severity([str(row.get("severity")) for row in region_reports])
    overall_risk = risk_from_reports(region_reports)
    summary_text = (
        f"{image_id}: {filtered_count}/{raw_count} retained reports, "
        f"main changes={main_change_types}, max severity={max_sev}, "
        f"GT coverage={gt_coverage:.3f}."
    )
    sample_summary = {
        "strategy": strategy.name,
        "image_id": image_id,
        "copyright_valid": rows[0].get("copyright_valid", "") if rows else "",
        "capsule_valid": rows[0].get("capsule_valid", "") if rows else "",
        "auth_status": rows[0].get("auth_status", "") if rows else "",
        "num_raw_regions": raw_count,
        "num_filtered_regions": filtered_count,
        "num_removed_regions": raw_count - filtered_count,
        "report_reduction_ratio": float((raw_count - filtered_count) / max(raw_count, 1)),
        "pred_area_retained": pred_area_retained,
        "gt_tamper_area_coverage": gt_coverage,
        "unknown_rate_before": unknown_before,
        "unknown_rate_after_filter": unknown_after,
        "num_valid_region_reports": filtered_count,
        "num_no_report_image": int(filtered_count == 0),
        "main_change_types": main_change_types,
        "max_severity": max_sev,
        "overall_risk": overall_risk,
        "severity_distribution": severity_dist,
        "change_type_distribution": change_dist,
        "report_summary": summary_text,
    }
    sample_summary.update(pix)
    return region_reports, sample_summary, tampered, pred_mask, aggregated_mask


def mean(rows: list[dict[str, Any]], field: str) -> float:
    vals = [safe_float(row.get(field)) for row in rows if row.get(field) not in ("", None)]
    return float(np.mean(vals)) if vals else 0.0


def summarize_strategy(
    strategy: Strategy,
    sample_rows: list[dict[str, Any]],
    region_rows: list[dict[str, Any]],
    f0_severity_dist: dict[str, int],
    f0_change_dist: dict[str, int],
) -> dict[str, Any]:
    total_raw = int(sum(safe_int(row.get("num_raw_regions")) for row in sample_rows))
    total_filtered = int(sum(safe_int(row.get("num_filtered_regions")) for row in sample_rows))
    unknown_after = sum(1 for row in region_rows if row.get("change_type") == "unknown") / max(total_filtered, 1)
    unknown_before = mean(sample_rows, "unknown_rate_before")
    no_report_images = int(sum(safe_int(row.get("num_no_report_image")) for row in sample_rows))
    severity_dist = distribution(region_rows, "severity")
    change_dist = distribution(region_rows, "change_type")
    mean_reports = total_filtered / max(len(sample_rows), 1)
    mean_pred_retained = mean(sample_rows, "pred_area_retained")
    mean_gt_coverage = mean(sample_rows, "gt_tamper_area_coverage")
    unknown_reduction_abs = unknown_before - unknown_after
    unknown_reduction_rel = unknown_reduction_abs / unknown_before if unknown_before else 0.0
    valid_report_coverage_rate = 1.0 - no_report_images / max(len(sample_rows), 1)
    meets_success = (
        unknown_after < 0.15
        and 3.0 <= mean_reports <= 5.0
        and no_report_images == 0
        and mean_pred_retained > 0.75
        and mean_gt_coverage > 0.85
    )
    anti_cheating_flag = (
        mean_pred_retained < 0.75
        or mean_gt_coverage < 0.85
        or no_report_images > 0
        or (unknown_after < 0.05 and mean_reports < 1.0)
    )
    compactness_score = max(0.0, 1.0 - abs(mean_reports - 4.0) / 16.0)
    score = (
        0.35 * (1.0 - unknown_after)
        + 0.30 * mean_gt_coverage
        + 0.15 * mean_pred_retained
        + 0.10 * valid_report_coverage_rate
        + 0.10 * compactness_score
    )
    return {
        "strategy": strategy.name,
        "total_images": len(sample_rows),
        "total_raw_regions": total_raw,
        "total_filtered_regions": total_filtered,
        "report_reduction_ratio": float((total_raw - total_filtered) / max(total_raw, 1)),
        "unknown_rate_before": unknown_before,
        "unknown_rate_after": unknown_after,
        "unknown_reduction_absolute": unknown_reduction_abs,
        "unknown_reduction_relative": unknown_reduction_rel,
        "mean_pred_area_retained": mean_pred_retained,
        "mean_gt_tamper_area_coverage": mean_gt_coverage,
        "mean_aggregated_iou": mean(sample_rows, "aggregated_iou"),
        "mean_aggregated_dice": mean(sample_rows, "aggregated_dice"),
        "no_report_images": no_report_images,
        "valid_report_coverage_rate": valid_report_coverage_rate,
        "mean_reports_per_image": mean_reports,
        "severity_distribution_stability": 1.0 - total_variation(f0_severity_dist, severity_dist),
        "change_type_distribution_stability": 1.0 - total_variation(f0_change_dist, change_dist),
        "meets_success_criteria": meets_success,
        "anti_cheating_flag": anti_cheating_flag,
        "recommended_by_tradeoff": False,
        "selection_score": float(score),
        "is_best": False,
    }


def choose_best_strategy(comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row
        for row in comparison_rows
        if not row["anti_cheating_flag"]
        and row["unknown_rate_after"] < 0.15
        and row["mean_pred_area_retained"] > 0.75
        and row["mean_gt_tamper_area_coverage"] > 0.85
        and row["no_report_images"] == 0
    ]
    if not viable:
        viable = [row for row in comparison_rows if not row["anti_cheating_flag"]] or comparison_rows
    f3_rows = [row for row in viable if row["strategy"] == "F3_merge_nearby_fragments" and row["mean_reports_per_image"] <= 8.0]
    if f3_rows:
        return max(f3_rows, key=lambda row: (row["selection_score"], -abs(row["mean_reports_per_image"] - 4.0)))
    compact_viable = [row for row in viable if 3.0 <= row["mean_reports_per_image"] <= 5.0]
    if compact_viable:
        return max(compact_viable, key=lambda row: (row["selection_score"], -row["unknown_rate_after"]))
    return max(viable, key=lambda row: (row["selection_score"], row["mean_gt_tamper_area_coverage"], -row["unknown_rate_after"]))


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (170, 170)) -> Image.Image:
    header_h = 30
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 8), title[:26], fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


def make_visual_row(
    strategy: Strategy,
    image_id: str,
    tampered: np.ndarray,
    pred_mask: np.ndarray,
    aggregated_mask: np.ndarray,
    sample_summary: dict[str, Any],
) -> Image.Image:
    panels = [
        title_panel(Image.fromarray(tampered), f"{strategy.name}"),
        title_panel(mask_image(pred_mask), "raw pred"),
        title_panel(mask_image(aggregated_mask), f"kept {sample_summary['num_filtered_regions']}"),
        title_panel(mask_image(np.logical_and(pred_mask, ~aggregated_mask)), "filtered"),
        title_panel(mask_image(aggregated_mask), f"GTcov {safe_float(sample_summary['gt_tamper_area_coverage']):.2f}"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def build_contact_sheet(rows: list[Image.Image], output_path: Path, max_items: int) -> dict[str, Any]:
    if len(rows) > max_items:
        indices = sorted({round(i * (len(rows) - 1) / (max_items - 1)) for i in range(max_items)})
        selected = [rows[idx] for idx in indices]
        sampling = "even"
    else:
        selected = rows
        sampling = "all"
    if not selected:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}
    resized = [row.convert("RGB").resize((850, 165), BICUBIC) for row in selected]
    canvas = Image.new("RGB", (resized[0].width, resized[0].height * len(resized)), "white")
    for idx, row in enumerate(resized):
        canvas.paste(row, (0, idx * row.height))
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(resized)}


def build_unknown_gallery(region_rows: list[dict[str, Any]], output_dir: Path, max_items: int) -> dict[str, Any]:
    unknown = [row for row in region_rows if row.get("change_type") == "unknown"]
    selected = unknown[:max_items]
    output_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Image.Image] = []
    for idx, row in enumerate(selected):
        bbox = row.get("bbox", [0, 0, 1, 1])
        image = Image.new("RGB", (90, 90), (240, 240, 240))
        draw = ImageDraw.Draw(image)
        draw.text((5, 5), str(row.get("image_id")), fill=(20, 20, 20))
        draw.text((5, 25), f"r{row.get('aggregated_region_id')}", fill=(20, 20, 20))
        draw.text((5, 45), str(row.get("size_bucket")), fill=(20, 20, 20))
        draw.text((5, 65), str(row.get("area_pixels")), fill=(20, 20, 20))
        panel = title_panel(image, f"{row.get('strategy')}")
        panel_path = output_dir / f"unknown_{idx:03d}_{row.get('image_id')}_r{row.get('aggregated_region_id')}.png"
        panel.save(panel_path)
        panels.append(panel)
    if not panels:
        return {"path": "", "items": 0}
    cols = 5
    rows_count = math.ceil(len(panels) / cols)
    canvas = Image.new("RGB", (panels[0].width * cols, panels[0].height * rows_count), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, ((idx % cols) * panel.width, (idx // cols) * panel.height))
    contact = output_dir / "filtered_unknown_case_gallery.png"
    canvas.save(contact)
    return {"path": str(contact.resolve()), "items": len(panels)}


def aggregated_examples(sample_rows: list[dict[str, Any]], region_rows: list[dict[str, Any]], best_strategy: str, max_items: int = 10) -> list[dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in region_rows:
        if row.get("strategy") == best_strategy:
            by_image[str(row.get("image_id"))].append(row)
    examples = []
    for sample in [row for row in sample_rows if row.get("strategy") == best_strategy][:max_items]:
        image_regions = by_image.get(str(sample.get("image_id")), [])
        examples.append(
            {
                "image_id": sample.get("image_id"),
                "strategy": best_strategy,
                "num_raw_regions": sample.get("num_raw_regions"),
                "num_filtered_regions": sample.get("num_filtered_regions"),
                "main_change_types": sample.get("main_change_types"),
                "max_severity": sample.get("max_severity"),
                "overall_risk": sample.get("overall_risk"),
                "gt_tamper_area_coverage": sample.get("gt_tamper_area_coverage"),
                "pred_area_retained": sample.get("pred_area_retained"),
                "report_summary": sample.get("report_summary"),
                "first_regions": image_regions[:3],
            }
        )
    return examples


def write_overview(path: Path, summary: dict[str, Any]) -> None:
    best = summary["best_strategy"]
    best_payload = summary["strategies"][best]
    lines = [
        "# Stage 8F Report-Level Region Aggregation",
        "",
        "Stage 8F is a report-level aggregation stage.",
        "It does not change watermark embedding, robust payload, or localization decoder.",
        "It addresses the main Stage 8E residual problem: tiny fragmented regions cause many unknown reports.",
        "It converts dense fragmented region reports into compact audit-level reports.",
        "Report filtering must preserve tamper coverage and not merely delete hard cases.",
        "",
        "## Best Strategy",
        "",
        f"- best_strategy: {best}",
        f"- unknown_rate_after: {best_payload['unknown_rate_after']}",
        f"- report_reduction_ratio: {best_payload['report_reduction_ratio']}",
        f"- mean_gt_tamper_area_coverage: {best_payload['mean_gt_tamper_area_coverage']}",
        f"- mean_pred_area_retained: {best_payload['mean_pred_area_retained']}",
        f"- no_report_images: {best_payload['no_report_images']}",
        f"- mean_reports_per_image: {best_payload['mean_reports_per_image']}",
        f"- recommended_by_tradeoff: {best_payload.get('recommended_by_tradeoff')}",
        f"- anti_cheating_flag: {best_payload.get('anti_cheating_flag')}",
        "",
        "No training, Stable Diffusion, SAM, CLIP, DINO, OmniGuard, payload change, or 128-bit payload is used.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = sorted(set(rows[0].keys())) if rows else []
    required = ["image_id", "region_id", "bbox", "area_pixels", "area_ratio", "stage8e_change_type", "severity"]
    optional = ["stage8e_confidence", "texture_score", "contrast_score", "size_bucket", "crop_path", "copyright_valid", "capsule_valid", "auth_status"]
    missing_required = [field for field in required if field not in available]
    missing_optional = [field for field in optional if field not in available]
    degraded_behaviors = []
    if "stage8e_confidence" in missing_optional:
        degraded_behaviors.append("pseudo-confidence derived from area/texture/contrast/change type/severity")
    if "size_bucket" in missing_optional:
        degraded_behaviors.append("size bucket derived from area_pixels")
    if "crop_path" in missing_optional:
        degraded_behaviors.append("tampered crop reconstructed from Stage 1 tampered.png and bbox")
    return {
        "available_columns": available,
        "missing_required_columns": missing_required,
        "missing_optional_columns": missing_optional,
        "degraded_behaviors": degraded_behaviors,
    }


def normalize_stage8e_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if not out.get("size_bucket"):
        out["size_bucket"] = s8e.bucket_size(safe_int(out.get("area_pixels")))
    if not out.get("stage8e_change_type"):
        out["stage8e_change_type"] = out.get("change_type", "unknown") or "unknown"
    if not out.get("stage8e_confidence"):
        out["stage8e_confidence"] = region_confidence(out)
    if not out.get("severity"):
        out["severity"] = "low"
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    stage8e_dir = (project_root / args.stage8e_dir).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    dirs = ensure_dirs(output_dir)

    raw_region_input = read_csv(stage8e_dir / "per_region_stage8e_reports.csv")
    schema_audit = inspect_schema(raw_region_input)
    if schema_audit["missing_required_columns"]:
        raise ValueError(f"Stage 8E region CSV is missing required columns: {schema_audit['missing_required_columns']}")
    region_input = [normalize_stage8e_row(row) for row in raw_region_input]
    stage8e_summary = json.loads((stage8e_dir / "stage8e_quality_summary.json").read_text(encoding="utf-8"))
    _unknown_analysis = json.loads((stage8e_dir / "stage8e_unknown_analysis.json").read_text(encoding="utf-8"))
    _examples = json.loads((stage8e_dir / "improved_report_examples.json").read_text(encoding="utf-8"))
    image_ids = sorted({str(row["image_id"]) for row in region_input})[: args.max_samples]
    rows_by_image: dict[str, list[dict[str, Any]]] = {
        image_id: [row for row in region_input if row.get("image_id") == image_id] for image_id in image_ids
    }

    all_region_rows: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    visual_rows: list[Image.Image] = []
    strategy_list = strategies()
    f0_severity_dist: dict[str, int] = {}
    f0_change_dist: dict[str, int] = {}
    strategy_sample_groups: dict[str, list[dict[str, Any]]] = {}
    strategy_region_groups: dict[str, list[dict[str, Any]]] = {}
    for strategy in strategy_list:
        strategy_samples: list[dict[str, Any]] = []
        strategy_regions: list[dict[str, Any]] = []
        for image_id, image_rows in rows_by_image.items():
            try:
                reports, sample_summary, tampered, pred_mask, aggregated_mask = process_strategy_for_image(
                    strategy, image_id, image_rows, stage1_dir, args
                )
                strategy_regions.extend(reports)
                strategy_samples.append(sample_summary)
                if strategy.name in {"F0_no_filter", "F1_tiny_filter_ar0p0005", "F3_merge_nearby_fragments", "F4_main_region_top5"}:
                    visual_rows.append(make_visual_row(strategy, image_id, tampered, pred_mask, aggregated_mask, sample_summary))
            except Exception as exc:
                failed.append(
                    {
                        "strategy": strategy.name,
                        "image_id": image_id,
                        "failure_stage": "process_strategy_for_image",
                        "error_message": str(exc),
                    }
                )
        if strategy.name == "F0_no_filter":
            f0_severity_dist = distribution(strategy_regions, "severity")
            f0_change_dist = distribution(strategy_regions, "change_type")
        strategy_sample_groups[strategy.name] = strategy_samples
        strategy_region_groups[strategy.name] = strategy_regions
        all_sample_rows.extend(strategy_samples)
        all_region_rows.extend(strategy_regions)
        print(f"Stage 8F strategy {strategy.name}: samples={len(strategy_samples)}, regions={len(strategy_regions)}, failed={len(failed)}", flush=True)

    comparison_rows = [
        summarize_strategy(strategy, strategy_sample_groups[strategy.name], strategy_region_groups[strategy.name], f0_severity_dist, f0_change_dist)
        for strategy in strategy_list
    ]
    best_row = choose_best_strategy(comparison_rows)
    for row in comparison_rows:
        row["is_best"] = row["strategy"] == best_row["strategy"]
        row["recommended_by_tradeoff"] = row["strategy"] == best_row["strategy"]

    contact_sheet = build_contact_sheet(visual_rows, output_dir / "stage8f_contact_sheet.png", args.contact_sheet_samples)
    unknown_gallery = build_unknown_gallery(strategy_region_groups[best_row["strategy"]], dirs["filtered_unknown_case_gallery"], args.unknown_gallery_samples)
    examples = aggregated_examples(all_sample_rows, all_region_rows, best_row["strategy"])

    write_csv(output_dir / "per_region_stage8f_reports.csv", all_region_rows, REGION_FIELDS)
    write_csv(output_dir / "per_sample_stage8f_summary.csv", all_sample_rows, SAMPLE_FIELDS)
    write_csv(output_dir / "stage8f_strategy_comparison.csv", comparison_rows, STRATEGY_FIELDS)
    (output_dir / "stage8f_failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "aggregated_report_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "stage": "stage8f_report_region_aggregation",
        "input_stage8e": {
            "total_region_reports": stage8e_summary.get("total_region_reports"),
            "unknown_change_type_rate_improved": stage8e_summary.get("unknown_change_type_rate_improved"),
        },
        "max_samples": args.max_samples,
        "strategies": {row["strategy"]: row for row in comparison_rows},
        "best_strategy": best_row["strategy"],
        "best_strategy_payload": best_row,
        "schema_audit": schema_audit,
        "failed_cases": len(failed),
        "contact_sheet": contact_sheet,
        "filtered_unknown_case_gallery": unknown_gallery,
        "constraints": {
            "no_training": True,
            "no_stable_diffusion": True,
            "no_sam_clip_dino": True,
            "no_omniguard_switch": True,
            "previous_stage_outputs_modified": False,
            "verification_does_not_read_original": True,
            "gt_mask_usage": ["evaluation_only"],
            "robust_payload_bits": 64,
            "robust_payload_changed": False,
            "no_128bit_payload_attempt": True,
        },
    }
    (output_dir / "stage8f_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_overview(output_dir / "stage8f_overview.md", summary)
    print(f"Saved strategy comparison: {output_dir / 'stage8f_strategy_comparison.csv'}")
    print(f"Saved summary: {output_dir / 'stage8f_summary.json'}")
    print(f"Saved overview: {output_dir / 'stage8f_overview.md'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
