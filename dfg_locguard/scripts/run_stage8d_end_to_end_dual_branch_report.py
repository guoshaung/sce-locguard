from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stage7c_non_oracle_semantic_template as s7c


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

STAGE8D_REGION_FIELDS = [
    "image_id",
    "region_id",
    "copyright_valid",
    "capsule_valid",
    "auth_status",
    "where",
    "bbox",
    "area_pixels",
    "area_ratio",
    "dominant_color",
    "texture_level",
    "contrast_level",
    "center_position",
    "before_semantics_from_capsule",
    "after_semantics_from_tampered_region",
    "change_type",
    "severity",
    "severity_score",
    "confidence",
    "payload_bits",
    "crop_path",
]

STAGE8D_SAMPLE_FIELDS = [
    "image_id",
    "status",
    "error_message",
    "tampered_path",
    "pred_mask_path",
    "gt_mask_path",
    "payload_source",
    "payload_bits",
    "copyright_valid",
    "capsule_valid",
    "auth_status",
    "bit_accuracy",
    "payload_recovery_accuracy",
    "capsule_recovery_accuracy",
    "auth_success",
    "iou",
    "dice",
    "precision",
    "recall",
    "mae",
    "report_coverage_rate",
    "unknown_change_type_rate",
    "num_pred_regions",
    "num_gt_regions",
    "num_valid_region_reports",
    "severity_distribution",
    "change_type_distribution",
    "sample_report_path",
]

CHANGE_TYPES = [
    "object_change",
    "attribute_change",
    "color_change",
    "texture_change",
    "risk_region_change",
    "unknown",
]

SEVERITY_LABELS = ["low", "medium", "high", "critical"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 8D end-to-end dual-branch semantic change report.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--stage1_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--stage8c_dir", default="dfg_locguard/outputs/stage8c_robust_payload_semantic_capsule_audit")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage8d_end_to_end_dual_branch_report")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--small_area_pixels", type=int, default=100)
    parser.add_argument("--tiny_area_pixels", type=int, default=25)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--payload_variant", default="P2_id_plus_compact_capsule_64")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "reports": output_dir / "reports",
        "tampered_region_crops": output_dir / "tampered_region_crops",
        "visualizations": output_dir / "visualizations",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def sample_dirs(stage1_dir: Path, max_samples: int) -> list[Path]:
    return sorted(path for path in stage1_dir.iterdir() if path.is_dir() and path.name.isdigit())[:max_samples]


def load_stage8c_rows(stage8c_dir: Path, payload_variant: str) -> dict[str, dict[str, Any]]:
    csv_path = stage8c_dir / "per_sample_stage8c_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Stage 8C per-sample metrics: {csv_path}")
    out: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("payload_variant") == payload_variant:
                out[str(row.get("image_id"))] = row
    return out


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)


def load_mask(path: Path, threshold: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((512, 512), NEAREST)
    return np.asarray(mask, dtype=np.uint8) > threshold


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except Exception:
        return default


def center_score(cx: float, cy: float, width: int, height: int) -> float:
    return s7c.center_score(cx, cy, width, height)


def severity_from_score(score: float) -> str:
    if score < 0.15:
        return "low"
    if score < 0.40:
        return "medium"
    if score < 0.65:
        return "high"
    return "critical"


def level_from_score(score: float) -> str:
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"


def color_name(rgb: np.ndarray) -> str:
    sample = np.uint8([[rgb]])
    hsv = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)[0, 0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if val < 45:
        return "dark"
    if sat < 35:
        return "gray_or_white"
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 25:
        return "orange"
    if hue < 40:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 125:
        return "blue"
    if hue < 150:
        return "purple"
    return "pink"


def position_name(cx: float, cy: float, width: int, height: int) -> str:
    x_zone = "left" if cx < width / 3 else "right" if cx > 2 * width / 3 else "center"
    y_zone = "top" if cy < height / 3 else "bottom" if cy > 2 * height / 3 else "middle"
    if x_zone == "center" and y_zone == "middle":
        return "center"
    return f"{y_zone}_{x_zone}"


def layout_name(layout_id: int) -> str:
    return {
        0: "upper_left_bright_region",
        1: "upper_right_bright_region",
        2: "lower_left_bright_region",
        3: "lower_right_bright_region",
    }.get(int(layout_id), "unknown_layout")


def risk_level_from_bits(value: int) -> str:
    if value < 64:
        return "low"
    if value < 128:
        return "medium"
    if value < 192:
        return "high"
    return "critical"


def texture_hint_from_attribute(attribute_bits: int) -> str:
    value = int(attribute_bits) & 0xF
    if value < 5:
        return "low"
    if value < 11:
        return "medium"
    return "high"


def decode_capsule(capsule: dict[str, Any]) -> dict[str, Any]:
    risk_bits = int(capsule.get("risk_bits", 0))
    attribute_bits = int(capsule.get("attribute_bits", 0))
    return {
        "scene_id": int(capsule.get("scene_id", 0)),
        "object_id": int(capsule.get("object_id", 0)),
        "attribute_bits": attribute_bits,
        "risk_bits": risk_bits,
        "layout_id": int(capsule.get("layout_id", 0)),
        "layout_hint": layout_name(int(capsule.get("layout_id", 0))),
        "risk_level": risk_level_from_bits(risk_bits),
        "texture_hint": texture_hint_from_attribute(attribute_bits),
        "source": "recovered_robust_p2_compact_capsule",
    }


def compute_feature_maps(tampered: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(tampered, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = s7c.normalize_01(np.sqrt(sx * sx + sy * sy))
    mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    mean_sq = cv2.GaussianBlur(gray * gray, (0, 0), sigmaX=3.0)
    contrast = s7c.normalize_01(np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)))
    return {"gray": gray, "texture": texture, "contrast": contrast}


def extract_region_reports(
    image_id: str,
    tampered: np.ndarray,
    pred_mask: np.ndarray,
    decoded_capsule: dict[str, Any],
    payload_row: dict[str, Any],
    dirs: dict[str, Path],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    height, width = pred_mask.shape
    image_area = float(height * width)
    feature_maps = compute_feature_maps(tampered)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
    severity_map = np.zeros((height, width), dtype=np.uint8)
    reports: list[dict[str, Any]] = []
    copyright_valid = safe_float(payload_row.get("bit_accuracy")) >= 0.999
    capsule_valid = safe_float(payload_row.get("semantic_capsule_recovery_accuracy")) >= 0.999
    auth_valid = safe_float(payload_row.get("auth_check_success")) >= 0.999
    auth_status = "valid" if auth_valid else "invalid"
    for idx in range(1, num_labels):
        region_mask = labels == idx
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[idx]
        area_ratio = float(area / image_area)
        area_score = float(np.clip(area_ratio / 0.20, 0.0, 1.0))
        c_score = center_score(float(cx), float(cy), width, height)
        texture_score = float(np.mean(feature_maps["texture"][region_mask]))
        contrast_score = float(np.mean(feature_maps["contrast"][region_mask]))
        severity_score = float(0.45 * area_score + 0.20 * c_score + 0.20 * texture_score + 0.15 * contrast_score)
        severity = severity_from_score(severity_score)
        severity_map[region_mask] = s7c.SEVERITY_INDEX[severity]

        pixels = tampered[region_mask]
        dominant = color_name(np.median(pixels, axis=0).astype(np.uint8))
        texture_level = level_from_score(texture_score)
        contrast_level = level_from_score(contrast_score)
        where = position_name(float(cx), float(cy), width, height)
        after_risk = severity if severity in {"high", "critical"} else "medium" if contrast_level == "high" else severity
        after_semantics = {
            "dominant_color": dominant,
            "texture_level": texture_level,
            "contrast_level": contrast_level,
            "center_position": where,
            "area_ratio": area_ratio,
            "risk_level": after_risk,
            "source": "tampered_region_features_only",
        }
        change_type = infer_change_type(decoded_capsule, after_semantics, area_score, severity)
        confidence = confidence_score(copyright_valid, capsule_valid, auth_valid, severity_score, area_score, change_type)

        crop_path = dirs["tampered_region_crops"] / f"{image_id}_region_{idx:03d}.png"
        crop = Image.fromarray(tampered[y : y + h, x : x + w], mode="RGB")
        crop.save(crop_path)
        reports.append(
            {
                "image_id": image_id,
                "region_id": int(idx),
                "copyright_valid": bool(copyright_valid),
                "capsule_valid": bool(capsule_valid),
                "auth_status": auth_status,
                "where": where,
                "bbox": [x, y, x + w, y + h],
                "area_pixels": area,
                "area_ratio": area_ratio,
                "dominant_color": dominant,
                "texture_level": texture_level,
                "contrast_level": contrast_level,
                "center_position": where,
                "before_semantics_from_capsule": decoded_capsule,
                "after_semantics_from_tampered_region": after_semantics,
                "change_type": change_type,
                "severity": severity,
                "predicted_severity_label": severity,
                "predicted_severity_index": int(s7c.SEVERITY_INDEX[severity]),
                "severity_score": severity_score,
                "confidence": confidence,
                "payload_bits": 64,
                "crop_path": str(crop_path.resolve()),
            }
        )
    return reports, severity_map


def infer_change_type(before: dict[str, Any], after: dict[str, Any], area_score: float, severity: str) -> str:
    if after["risk_level"] != before.get("risk_level") and severity in {"high", "critical"}:
        return "risk_region_change"
    if after["texture_level"] != before.get("texture_hint") and after["texture_level"] == "high":
        return "texture_change"
    if after["dominant_color"] not in {"gray_or_white", "dark"} and after["contrast_level"] in {"medium", "high"}:
        return "color_change"
    if area_score > 0.25 and after["center_position"] not in before.get("layout_hint", ""):
        return "object_change"
    if after["contrast_level"] in {"medium", "high"}:
        return "attribute_change"
    return "unknown"


def confidence_score(
    copyright_valid: bool,
    capsule_valid: bool,
    auth_valid: bool,
    severity_score: float,
    area_score: float,
    change_type: str,
) -> float:
    base = 0.35 + 0.25 * float(copyright_valid) + 0.20 * float(capsule_valid) + 0.10 * float(auth_valid)
    signal = 0.05 * min(1.0, severity_score / 0.65) + 0.05 * min(1.0, area_score / 0.20)
    if change_type == "unknown":
        base -= 0.20
    return float(np.clip(base + signal, 0.0, 1.0))


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


def error_map_image(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    out = np.zeros((*gt_bool.shape, 3), dtype=np.uint8)
    out[np.logical_and(pred_bool, gt_bool)] = [255, 255, 255]
    out[np.logical_and(pred_bool, ~gt_bool)] = [255, 80, 80]
    out[np.logical_and(~pred_bool, gt_bool)] = [80, 160, 255]
    return Image.fromarray(out, mode="RGB")


def severity_overlay(tampered: np.ndarray, severity_map: np.ndarray, reports: list[dict[str, Any]]) -> Image.Image:
    image = s7c.severity_overlay(Image.fromarray(tampered), severity_map, reports)
    return image


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (170, 170)) -> Image.Image:
    header_h = 28
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title[:26], fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def save_visualization(
    image_id: str,
    tampered: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    severity_map: np.ndarray,
    reports: list[dict[str, Any]],
    sample_summary: dict[str, Any],
    dirs: dict[str, Path],
) -> Path:
    overlay = severity_overlay(tampered, severity_map, reports)
    panels = [
        title_panel(Image.fromarray(tampered), image_id),
        title_panel(mask_image(pred_mask), f"IoU {safe_float(sample_summary.get('iou')):.3f}"),
        title_panel(mask_image(gt_mask), "GT eval only"),
        title_panel(error_map_image(gt_mask, pred_mask), "loc error"),
        title_panel(overlay, f"reports {len(reports)}"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    out_path = dirs["visualizations"] / f"{image_id}_stage8d_vis.png"
    canvas.save(out_path)
    return out_path


def write_sample_report(path: Path, sample_summary: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    lines = [
        f"# Stage 8D Dual-Branch Report: {sample_summary['image_id']}",
        "",
        "- fragile_branch: EditGuard predicted tamper mask",
        "- robust_branch: recovered P2 64-bit copyright ID + compact semantic capsule + checksum",
        "- verification_original_access: false",
        f"- auth_status: {sample_summary['auth_status']}",
        f"- copyright_valid: {sample_summary['copyright_valid']}",
        f"- capsule_valid: {sample_summary['capsule_valid']}",
        "",
        "## Localization",
        "",
        f"- IoU: {sample_summary['iou']}",
        f"- Dice: {sample_summary['dice']}",
        f"- Precision: {sample_summary['precision']}",
        f"- Recall: {sample_summary['recall']}",
        f"- MAE: {sample_summary['mae']}",
        "",
        "## Region Reports",
        "",
    ]
    for report in reports[:50]:
        lines.extend(
            [
                f"### Region {report['region_id']}",
                "",
                f"- where: {report['where']}",
                f"- change_type: {report['change_type']}",
                f"- severity: {report['severity']}",
                f"- confidence: {report['confidence']}",
                f"- before_semantics_from_capsule: `{json.dumps(report['before_semantics_from_capsule'], ensure_ascii=False)}`",
                f"- after_semantics_from_tampered_region: `{json.dumps(report['after_semantics_from_tampered_region'], ensure_ascii=False)}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def process_sample(
    sample_dir: Path,
    payload_row: dict[str, Any],
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    image_id = sample_dir.name
    tampered_path = sample_dir / "tampered.png"
    pred_mask_path = sample_dir / "editguard_pred_mask.png"
    gt_mask_path = sample_dir / "gt_mask.png"
    for required in [tampered_path, pred_mask_path, gt_mask_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing verification/evaluation input: {required}")
    if payload_row.get("status") != "ok":
        raise ValueError(f"Stage 8C P2 payload is not valid for {image_id}: {payload_row.get('status')}")

    # Verification-side inputs: tampered image, fragile pred mask, recovered robust payload.
    # This function intentionally does not read original.png.
    tampered = load_rgb(tampered_path)
    pred_mask = load_mask(pred_mask_path, args.mask_threshold)
    gt_mask = load_mask(gt_mask_path, args.mask_threshold)
    capsule = json.loads(payload_row.get("semantic_capsule_json") or "{}")
    decoded_capsule = decode_capsule(capsule)
    reports, severity_map = extract_region_reports(image_id, tampered, pred_mask, decoded_capsule, payload_row, dirs)
    pixel = s7c.pixel_metrics(gt_mask, pred_mask)
    num_pred, _pred_labels, _pred_stats, _ = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
    num_gt, _gt_labels, _gt_stats, _ = cv2.connectedComponentsWithStats(gt_mask.astype(np.uint8), connectivity=8)
    change_counts = Counter(report["change_type"] for report in reports)
    severity_counts = Counter(report["severity"] for report in reports)
    valid_auth = safe_float(payload_row.get("auth_check_success")) >= 0.999
    copyright_valid = safe_float(payload_row.get("bit_accuracy")) >= 0.999
    capsule_valid = safe_float(payload_row.get("semantic_capsule_recovery_accuracy")) >= 0.999
    valid_region_reports = len(reports) if (valid_auth and copyright_valid and capsule_valid) else 0
    sample_summary: dict[str, Any] = {
        "image_id": image_id,
        "status": "ok",
        "error_message": "",
        "tampered_path": str(tampered_path.resolve()),
        "pred_mask_path": str(pred_mask_path.resolve()),
        "gt_mask_path": str(gt_mask_path.resolve()),
        "payload_source": "stage8c_recovered_P2_id_plus_compact_capsule_64",
        "payload_bits": 64,
        "copyright_valid": bool(copyright_valid),
        "capsule_valid": bool(capsule_valid),
        "auth_status": "valid" if valid_auth else "invalid",
        "bit_accuracy": safe_float(payload_row.get("bit_accuracy")),
        "payload_recovery_accuracy": safe_float(payload_row.get("payload_recovery_accuracy")),
        "capsule_recovery_accuracy": safe_float(payload_row.get("semantic_capsule_recovery_accuracy")),
        "auth_success": safe_float(payload_row.get("auth_check_success")),
        "report_coverage_rate": s7c.safe_div(len(reports), max(num_pred - 1, 1)),
        "unknown_change_type_rate": s7c.safe_div(change_counts.get("unknown", 0), max(len(reports), 1)),
        "num_pred_regions": int(max(num_pred - 1, 0)),
        "num_gt_regions": int(max(num_gt - 1, 0)),
        "num_valid_region_reports": int(valid_region_reports),
        "severity_distribution": json.dumps(dict(severity_counts), sort_keys=True),
        "change_type_distribution": json.dumps(dict(change_counts), sort_keys=True),
    }
    sample_summary.update(pixel)
    sample_report_path = dirs["reports"] / f"{image_id}_stage8d_report.md"
    sample_regions_path = dirs["reports"] / f"{image_id}_stage8d_regions.json"
    sample_summary["sample_report_path"] = str(sample_report_path.resolve())
    sample_regions_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sample_report(sample_report_path, sample_summary, reports)
    vis_path = save_visualization(image_id, tampered, pred_mask, gt_mask, severity_map, reports, sample_summary, dirs)
    return reports, sample_summary, vis_path


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serializable: dict[str, Any] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                serializable[field] = value
            writer.writerow(serializable)


def mean(values: list[Any]) -> float:
    nums: list[float] = []
    for value in values:
        if value in ("", None):
            continue
        try:
            nums.append(float(value))
        except Exception:
            continue
    return float(np.mean(nums)) if nums else 0.0


def build_contact_sheet(vis_paths: list[Path], output_path: Path, max_items: int) -> dict[str, Any]:
    existing = [path for path in vis_paths if path.exists()]
    if len(existing) > max_items:
        indices = sorted({round(i * (len(existing) - 1) / (max_items - 1)) for i in range(max_items)})
        selected = [existing[idx] for idx in indices]
        sampling = "even"
    else:
        selected = existing
        sampling = "all"
    rows = [Image.open(path).convert("RGB").resize((850, 165), BICUBIC) for path in selected]
    if not rows:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}
    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(rows)}


def payload_examples(sample_summaries: list[dict[str, Any]], region_rows: list[dict[str, Any]], max_items: int = 5) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in region_rows:
        by_image[row["image_id"]].append(row)
    for sample in sample_summaries[:max_items]:
        regions = by_image.get(sample["image_id"], [])
        examples.append(
            {
                "image_id": sample["image_id"],
                "auth_status": sample.get("auth_status", ""),
                "copyright_valid": sample.get("copyright_valid", False),
                "capsule_valid": sample.get("capsule_valid", False),
                "payload_bits": sample.get("payload_bits", ""),
                "first_region_report": regions[0] if regions else None,
            }
        )
    return examples


def write_overview(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 8D End-to-End Dual-Branch Semantic Change Report",
        "",
        "This stage uses both branches of EditGuard.",
        "Fragile branch provides tamper localization.",
        "Robust branch provides copyright ID and compact semantic capsule.",
        "Verification does not access the original image.",
        "The semantic capsule is recovered from robust watermark payload.",
        "The output is a localized semantic change report.",
        "",
        "## Results",
        "",
        f"- total_samples: {summary['total_samples']}",
        f"- evaluated_samples: {summary['evaluated_samples']}",
        f"- failed_samples: {summary['failed_samples']}",
        f"- total_region_reports: {summary['total_region_reports']}",
        f"- mean_iou: {summary['mean_iou']}",
        f"- mean_dice: {summary['mean_dice']}",
        f"- mean_bit_accuracy: {summary['mean_bit_accuracy']}",
        f"- mean_payload_recovery_accuracy: {summary['mean_payload_recovery_accuracy']}",
        f"- mean_capsule_recovery_accuracy: {summary['mean_capsule_recovery_accuracy']}",
        f"- mean_auth_success: {summary['mean_auth_success']}",
        f"- mean_report_coverage_rate: {summary['mean_report_coverage_rate']}",
        f"- mean_unknown_change_type_rate: {summary['mean_unknown_change_type_rate']}",
        "",
        "## Verification Inputs",
        "",
        "- tampered image",
        "- EditGuard fragile-branch predicted tamper mask",
        "- Stage 8C recovered P2 64-bit compact semantic capsule",
        "- recovered copyright/auth fields",
        "",
        "GT masks are used only for evaluation metrics. No Stable Diffusion, SAM, CLIP, DINO, OmniGuard, or training is used.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve()
    stage8c_dir = (project_root / args.stage8c_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    dirs = ensure_dirs(output_dir)
    p2_rows = load_stage8c_rows(stage8c_dir, args.payload_variant)
    samples = sample_dirs(stage1_dir, args.max_samples)
    if not samples:
        raise FileNotFoundError(f"No Stage 1 samples found under {stage1_dir}")

    region_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    vis_paths: list[Path] = []
    for index, sample_dir in enumerate(samples, start=1):
        image_id = sample_dir.name
        try:
            if image_id not in p2_rows:
                raise FileNotFoundError(f"Missing Stage 8C P2 recovered payload row for image_id={image_id}")
            reports, sample_summary, vis_path = process_sample(sample_dir, p2_rows[image_id], dirs, args)
            region_rows.extend(reports)
            sample_rows.append(sample_summary)
            vis_paths.append(vis_path)
        except Exception as exc:
            failed.append(
                {
                    "image_id": image_id,
                    "failure_stage": "stage8d_process_sample",
                    "error_message": str(exc),
                }
            )
            sample_rows.append(
                {
                    "image_id": image_id,
                    "status": "failed",
                    "error_message": str(exc),
                }
            )
        if index % 10 == 0 or index == len(samples):
            print(f"Stage 8D progress: {index}/{len(samples)} samples, failed={len(failed)}", flush=True)

    write_csv(output_dir / "per_region_stage8d_reports.csv", region_rows, STAGE8D_REGION_FIELDS)
    write_csv(output_dir / "per_sample_stage8d_summary.csv", sample_rows, STAGE8D_SAMPLE_FIELDS)
    (output_dir / "stage8d_failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "payload_decoding_examples.json").write_text(
        json.dumps(payload_examples(sample_rows, region_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contact_sheet = build_contact_sheet(vis_paths, output_dir / "stage8d_contact_sheet.png", args.contact_sheet_samples)
    ok_samples = [row for row in sample_rows if row.get("status") == "ok"]
    severity_counts = Counter(row["severity"] for row in region_rows)
    change_counts = Counter(row["change_type"] for row in region_rows)
    summary = {
        "stage": "stage8d_end_to_end_dual_branch_report",
        "total_samples": len(samples),
        "evaluated_samples": len(ok_samples),
        "failed_samples": len(failed),
        "total_region_reports": len(region_rows),
        "valid_region_reports": int(sum(int(row.get("num_valid_region_reports", 0)) for row in ok_samples)),
        "payload_variant": args.payload_variant,
        "payload_bits": 64,
        "protection_side_capsule_source": "Stage 8C P2 compact semantic capsule generated from protection-side original image and recovered by robust branch",
        "verification_original_access": False,
        "uses_both_editguard_branches": True,
        "fragile_branch_output": "predicted tamper mask M_pred",
        "robust_branch_output": "copyright_id + compact semantic capsule + checksum/auth from 64-bit payload",
        "gt_mask_usage": ["evaluation_only"],
        "mean_iou": mean([row.get("iou") for row in ok_samples]),
        "mean_dice": mean([row.get("dice") for row in ok_samples]),
        "mean_precision": mean([row.get("precision") for row in ok_samples]),
        "mean_recall": mean([row.get("recall") for row in ok_samples]),
        "mean_mae": mean([row.get("mae") for row in ok_samples]),
        "mean_bit_accuracy": mean([row.get("bit_accuracy") for row in ok_samples]),
        "mean_payload_recovery_accuracy": mean([row.get("payload_recovery_accuracy") for row in ok_samples]),
        "mean_capsule_recovery_accuracy": mean([row.get("capsule_recovery_accuracy") for row in ok_samples]),
        "mean_auth_success": mean([row.get("auth_success") for row in ok_samples]),
        "mean_report_coverage_rate": mean([row.get("report_coverage_rate") for row in ok_samples]),
        "mean_unknown_change_type_rate": mean([row.get("unknown_change_type_rate") for row in ok_samples]),
        "severity_distribution": dict(severity_counts),
        "change_type_distribution": dict(change_counts),
        "contact_sheet": contact_sheet,
        "constraints": {
            "no_training": True,
            "no_stable_diffusion": True,
            "no_sam_clip_dino": True,
            "no_omniguard_switch": True,
            "previous_stage_outputs_modified": False,
            "verification_does_not_read_original": True,
            "no_128bit_payload_attempt": True,
        },
    }
    (output_dir / "stage8d_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_overview(output_dir / "stage8d_overview.md", summary)
    print(f"Saved per-region reports: {output_dir / 'per_region_stage8d_reports.csv'}")
    print(f"Saved summary: {output_dir / 'stage8d_summary.json'}")
    print(f"Saved overview: {output_dir / 'stage8d_overview.md'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
