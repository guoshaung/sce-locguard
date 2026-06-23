from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
NEAREST = getattr(Image, "Resampling", Image).NEAREST

CHANGE_TYPES = [
    "color_change",
    "texture_change",
    "structure_change",
    "removal_or_occlusion",
    "risk_region_change",
    "unknown",
]

REGION_FIELDS = [
    "image_id",
    "region_id",
    "auth_status",
    "copyright_valid",
    "capsule_valid",
    "payload_bits",
    "bbox",
    "area_pixels",
    "area_ratio",
    "size_bucket",
    "center_score",
    "texture_score",
    "contrast_score",
    "brightness_score",
    "edge_density",
    "contour_complexity",
    "texture_bucket",
    "confidence_bucket",
    "fragmentation_indicator",
    "severity",
    "severity_score",
    "stage8d_change_type",
    "stage8e_level1_change_state",
    "stage8e_change_type",
    "stage8e_reason",
    "stage8d_confidence",
    "stage8e_confidence",
    "before_semantics_from_capsule",
    "after_semantics_from_tampered_region",
    "crop_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 8E report quality audit and unknown-type reduction.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--stage8d_dir", default="dfg_locguard/outputs/stage8d_end_to_end_dual_branch_report")
    parser.add_argument("--stage1_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage8e_report_quality_audit")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--unknown_gallery_samples", type=int, default=80)
    parser.add_argument("--use_vlm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "unknown_case_gallery": output_dir / "unknown_case_gallery",
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
            out: dict[str, Any] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                out[field] = value
            writer.writerow(out)


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "valid"}


def parse_json_field(value: Any, default: Any) -> Any:
    if value in ("", None):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def center_score_from_bbox(bbox: list[int], width: int = 512, height: int = 512) -> float:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    image_cx = (width - 1) / 2.0
    image_cy = (height - 1) / 2.0
    max_dist = math.sqrt(image_cx * image_cx + image_cy * image_cy)
    dist = math.sqrt((cx - image_cx) ** 2 + (cy - image_cy) ** 2)
    return float(np.clip(1.0 - dist / max(max_dist, 1e-8), 0.0, 1.0))


def bucket_size(area_pixels: int) -> str:
    if area_pixels < 25:
        return "tiny"
    if area_pixels < 100:
        return "small"
    if area_pixels < 1000:
        return "medium"
    return "large"


def bucket_score(score: float) -> str:
    if score < 0.33:
        return "low"
    if score < 0.66:
        return "medium"
    return "high"


def bucket_confidence(score: float) -> str:
    if score < 0.50:
        return "low"
    if score < 0.75:
        return "medium"
    return "high"


def load_crop(row: dict[str, Any], stage1_dir: Path) -> np.ndarray:
    crop_path = Path(str(row.get("crop_path", "")))
    if crop_path.exists():
        return np.asarray(Image.open(crop_path).convert("RGB"), dtype=np.uint8)
    image_id = str(row["image_id"])
    bbox = parse_json_field(row.get("bbox"), [0, 0, 1, 1])
    tampered_path = stage1_dir / image_id / "tampered.png"
    tampered = np.asarray(Image.open(tampered_path).convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return tampered[y1:y2, x1:x2]


def region_image_features(crop: np.ndarray) -> dict[str, float]:
    if crop.size == 0:
        return {
            "texture_score": 0.0,
            "contrast_score": 0.0,
            "brightness_score": 0.0,
            "edge_density": 0.0,
            "contour_complexity": 0.0,
            "red_dominance": 0.0,
            "blue_dominance": 0.0,
            "saturation": 0.0,
        }
    rgb = crop.astype(np.float32) / 255.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sx * sx + sy * sy)
    grad_norm = normalize_01(grad)
    contrast = float(np.std(gray))
    brightness = float(np.mean(gray))
    edge_density = float(np.mean(grad_norm > 0.45))
    binary = (grad_norm > 0.45).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_perim = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    contour_complexity = float(np.clip(contour_perim / max(float(crop.shape[0] * crop.shape[1]), 1.0), 0.0, 1.0))
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
    mean_rgb = np.mean(rgb.reshape(-1, 3), axis=0)
    red_dominance = float(max(0.0, mean_rgb[0] - max(mean_rgb[1], mean_rgb[2])))
    blue_dominance = float(max(0.0, mean_rgb[2] - max(mean_rgb[0], mean_rgb[1])))
    return {
        "texture_score": float(np.mean(grad_norm)),
        "contrast_score": float(np.clip(contrast * 3.0, 0.0, 1.0)),
        "brightness_score": brightness,
        "edge_density": edge_density,
        "contour_complexity": contour_complexity,
        "red_dominance": red_dominance,
        "blue_dominance": blue_dominance,
        "saturation": saturation,
    }


def risk_level_to_score(label: str) -> float:
    return {"low": 0.20, "medium": 0.45, "high": 0.70, "critical": 0.95}.get(str(label), 0.20)


def improved_change_type(row: dict[str, Any], features: dict[str, float]) -> tuple[str, str, float, str]:
    before = parse_json_field(row.get("before_semantics_from_capsule"), {})
    after = parse_json_field(row.get("after_semantics_from_tampered_region"), {})
    area_ratio = safe_float(row.get("area_ratio"))
    area_score = float(np.clip(area_ratio / 0.20, 0.0, 1.0))
    center = safe_float(row.get("center_score"), 0.0)
    severity = str(row.get("severity", "low"))
    capsule_risk = str(before.get("risk_level", "low"))
    capsule_texture = str(before.get("texture_hint", "low"))
    dominant_color = str(after.get("dominant_color", row.get("dominant_color", "")))
    after_contrast = str(after.get("contrast_level", row.get("contrast_level", "low")))
    after_texture = str(after.get("texture_level", row.get("texture_level", "low")))
    brightness = features["brightness_score"]
    texture = features["texture_score"]
    contrast = features["contrast_score"]
    edge_density = features["edge_density"]
    complexity = features["contour_complexity"]
    color_signal = max(features["red_dominance"], features["blue_dominance"], features["saturation"] - 0.35)
    capsule_risk_score = risk_level_to_score(capsule_risk)

    if color_signal > 0.20 or dominant_color in {"red", "blue", "orange", "yellow", "green", "purple", "pink"} and contrast > 0.18:
        return "changed", "color_change", confidence_from_signals(0.72, color_signal, contrast, area_score), "dominant saturated color / color proxy"
    if brightness < 0.18 and contrast < 0.22 and area_score > 0.01:
        return "changed", "removal_or_occlusion", confidence_from_signals(0.66, 1.0 - brightness, area_score, 0.0), "dark low-contrast region suggests occlusion/removal"
    if edge_density > 0.34 or complexity > 0.20:
        return "changed", "structure_change", confidence_from_signals(0.64, edge_density, complexity, area_score), "high edge density or contour complexity"
    if texture > 0.34 or (after_texture == "high" and capsule_texture != "high"):
        return "changed", "texture_change", confidence_from_signals(0.62, texture, contrast, edge_density), "high texture/gradient proxy"
    if capsule_risk_score >= 0.70 and (center > 0.58 or area_score > 0.08 or severity in {"high", "critical"}):
        return "changed", "risk_region_change", confidence_from_signals(0.70, capsule_risk_score, center, area_score), "capsule risk tag with central/large predicted region"
    if after_contrast in {"medium", "high"} or contrast > 0.22:
        return "changed", "texture_change", confidence_from_signals(0.56, contrast, texture, area_score), "medium/high contrast proxy"
    return "uncertain", "unknown", confidence_from_signals(0.35, area_score, texture, contrast), "weak visual proxy signals"


def confidence_from_signals(base: float, *signals: float) -> float:
    signal = float(np.mean([np.clip(v, 0.0, 1.0) for v in signals])) if signals else 0.0
    return float(np.clip(base + 0.25 * signal, 0.0, 1.0))


def fragmentation_indicator(row: dict[str, Any], sample_lookup: dict[str, dict[str, Any]]) -> float:
    sample = sample_lookup.get(str(row["image_id"]), {})
    pred_regions = safe_float(sample.get("num_pred_regions"), 1.0)
    gt_regions = max(safe_float(sample.get("num_gt_regions"), 1.0), 1.0)
    area = safe_float(row.get("area_pixels"))
    size_penalty = 1.0 if area < 25 else 0.7 if area < 100 else 0.25 if area < 1000 else 0.0
    return float(np.clip(0.5 * min(pred_regions / (gt_regions * 20.0), 1.0) + 0.5 * size_penalty, 0.0, 1.0))


def process_region_row(
    row: dict[str, Any],
    sample_lookup: dict[str, dict[str, Any]],
    stage1_dir: Path,
) -> dict[str, Any]:
    bbox = parse_json_field(row.get("bbox"), [0, 0, 1, 1])
    crop = load_crop(row, stage1_dir)
    features = region_image_features(crop)
    center = center_score_from_bbox([int(v) for v in bbox])
    work = dict(row)
    work["center_score"] = center
    state, new_type, new_conf, reason = improved_change_type(work, features)
    area_pixels = safe_int(row.get("area_pixels"))
    out = {
        **row,
        "area_pixels": area_pixels,
        "area_ratio": safe_float(row.get("area_ratio")),
        "size_bucket": bucket_size(area_pixels),
        "center_score": center,
        "texture_score": features["texture_score"],
        "contrast_score": features["contrast_score"],
        "brightness_score": features["brightness_score"],
        "edge_density": features["edge_density"],
        "contour_complexity": features["contour_complexity"],
        "texture_bucket": bucket_score(features["texture_score"]),
        "confidence_bucket": bucket_confidence(safe_float(row.get("confidence"))),
        "fragmentation_indicator": fragmentation_indicator(row, sample_lookup),
        "stage8d_change_type": row.get("change_type", ""),
        "stage8e_level1_change_state": state,
        "stage8e_change_type": new_type,
        "stage8e_reason": reason,
        "stage8d_confidence": safe_float(row.get("confidence")),
        "stage8e_confidence": new_conf,
    }
    return out


def unknown_rate(rows: list[dict[str, Any]], field: str = "stage8d_change_type") -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if row.get(field) == "unknown") / len(rows))


def bucket_unknown_rates(rows: list[dict[str, Any]], bucket_field: str, type_field: str = "stage8d_change_type") -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(bucket_field, ""))].append(row)
    return {
        bucket: {
            "count": len(items),
            "unknown_count": int(sum(1 for item in items if item.get(type_field) == "unknown")),
            "unknown_rate": unknown_rate(items, type_field),
        }
        for bucket, items in sorted(groups.items())
    }


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "")) for row in rows))


def summarize_quality(stage8d_summary: dict[str, Any], sample_rows: list[dict[str, Any]], region_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_reports = [row for row in region_rows if parse_bool(row.get("copyright_valid")) and parse_bool(row.get("capsule_valid")) and row.get("auth_status") == "valid"]
    original_unknown = unknown_rate(region_rows, "stage8d_change_type")
    improved_unknown = unknown_rate(region_rows, "stage8e_change_type")
    return {
        "stage": "stage8e_report_quality_audit",
        "input_stage8d_summary": {
            "evaluated_samples": stage8d_summary.get("evaluated_samples"),
            "total_region_reports": stage8d_summary.get("total_region_reports"),
            "unknown_change_type_rate": stage8d_summary.get("mean_unknown_change_type_rate"),
        },
        "total_region_reports": len(region_rows),
        "valid_report_count": len(valid_reports),
        "unknown_change_type_count_original": int(sum(1 for row in region_rows if row.get("stage8d_change_type") == "unknown")),
        "unknown_change_type_rate_original": original_unknown,
        "unknown_change_type_count_improved": int(sum(1 for row in region_rows if row.get("stage8e_change_type") == "unknown")),
        "unknown_change_type_rate_improved": improved_unknown,
        "unknown_rate_reduction_abs": original_unknown - improved_unknown,
        "unknown_rate_reduction_relative": (original_unknown - improved_unknown) / original_unknown if original_unknown else 0.0,
        "change_type_distribution_original": distribution(region_rows, "stage8d_change_type"),
        "change_type_distribution_improved": distribution(region_rows, "stage8e_change_type"),
        "severity_distribution": distribution(region_rows, "severity"),
        "confidence_distribution": distribution(region_rows, "confidence_bucket"),
        "auth_valid_rate": float(np.mean([row.get("auth_status") == "valid" for row in region_rows])) if region_rows else 0.0,
        "capsule_valid_rate": float(np.mean([parse_bool(row.get("capsule_valid")) for row in region_rows])) if region_rows else 0.0,
        "copyright_valid_rate": float(np.mean([parse_bool(row.get("copyright_valid")) for row in region_rows])) if region_rows else 0.0,
        "report_coverage_rate": float(np.mean([safe_float(row.get("report_coverage_rate"), 0.0) for row in sample_rows])) if sample_rows else 0.0,
        "auth_success": float(np.mean([safe_float(row.get("auth_success"), 0.0) for row in sample_rows])) if sample_rows else 0.0,
        "capsule_validity": float(np.mean([safe_float(row.get("capsule_recovery_accuracy"), 0.0) for row in sample_rows])) if sample_rows else 0.0,
    }


def unknown_analysis(region_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "unknown_rate_by_region_size_bucket": bucket_unknown_rates(region_rows, "size_bucket"),
        "unknown_rate_by_severity_bucket": bucket_unknown_rates(region_rows, "severity"),
        "unknown_rate_by_texture_bucket": bucket_unknown_rates(region_rows, "texture_bucket"),
        "unknown_rate_by_confidence_bucket": bucket_unknown_rates(region_rows, "confidence_bucket"),
        "improved_unknown_rate_by_region_size_bucket": bucket_unknown_rates(region_rows, "size_bucket", "stage8e_change_type"),
        "improved_unknown_rate_by_severity_bucket": bucket_unknown_rates(region_rows, "severity", "stage8e_change_type"),
        "improved_unknown_rate_by_texture_bucket": bucket_unknown_rates(region_rows, "texture_bucket", "stage8e_change_type"),
        "improved_unknown_rate_by_confidence_bucket": bucket_unknown_rates(region_rows, "confidence_bucket", "stage8e_change_type"),
    }


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (150, 150)) -> Image.Image:
    header_h = 30
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 8), title[:24], fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def make_case_panel(row: dict[str, Any], title: str) -> Image.Image:
    crop_path = Path(str(row.get("crop_path", "")))
    if crop_path.exists():
        image = Image.open(crop_path).convert("RGB")
    else:
        image = Image.new("RGB", (64, 64), (240, 240, 240))
    label = f"{title} {row.get('stage8e_change_type')} {safe_float(row.get('stage8e_confidence')):.2f}"
    return title_panel(image, label)


def build_gallery(rows: list[dict[str, Any]], output_dir: Path, max_items: int) -> dict[str, Any]:
    unknown_original = [row for row in rows if row.get("stage8d_change_type") == "unknown"]
    selected = unknown_original[:max_items]
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for idx, row in enumerate(selected):
        panel = make_case_panel(row, f"{row.get('image_id')}-{row.get('region_id')}")
        panel_path = output_dir / f"unknown_{idx:03d}_{row.get('image_id')}_r{row.get('region_id')}.png"
        panel.save(panel_path)
        panels.append(panel)
    if not panels:
        return {"gallery_path": "", "items": 0}
    cols = 5
    rows_count = math.ceil(len(panels) / cols)
    canvas = Image.new("RGB", (panels[0].width * cols, panels[0].height * rows_count), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, ((idx % cols) * panel.width, (idx // cols) * panel.height))
    contact_path = output_dir / "unknown_case_gallery_contact_sheet.png"
    canvas.save(contact_path)
    return {"gallery_path": str(contact_path.resolve()), "items": len(panels)}


def build_contact_sheet(rows: list[dict[str, Any]], output_path: Path, max_items: int) -> dict[str, Any]:
    selected = rows
    if len(selected) > max_items:
        indices = sorted({round(i * (len(selected) - 1) / (max_items - 1)) for i in range(max_items)})
        selected = [selected[idx] for idx in indices]
        sampling = "even"
    else:
        sampling = "all"
    panels = [make_case_panel(row, str(row.get("image_id"))) for row in selected]
    if not panels:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}
    cols = 5
    rows_count = math.ceil(len(panels) / cols)
    canvas = Image.new("RGB", (panels[0].width * cols, panels[0].height * rows_count), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, ((idx % cols) * panel.width, (idx // cols) * panel.height))
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(panels)}


def improved_examples(region_rows: list[dict[str, Any]], max_items: int = 20) -> list[dict[str, Any]]:
    changed = [
        row
        for row in region_rows
        if row.get("stage8d_change_type") == "unknown" and row.get("stage8e_change_type") != "unknown"
    ]
    out = []
    for row in changed[:max_items]:
        out.append(
            {
                "image_id": row.get("image_id"),
                "region_id": row.get("region_id"),
                "stage8d_change_type": row.get("stage8d_change_type"),
                "stage8e_change_type": row.get("stage8e_change_type"),
                "stage8e_reason": row.get("stage8e_reason"),
                "stage8e_confidence": row.get("stage8e_confidence"),
                "severity": row.get("severity"),
                "crop_path": row.get("crop_path"),
            }
        )
    return out


def write_overview(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 8E Report Quality Audit",
        "",
        "Stage 8D already validates the dual-watermark pipeline.",
        "Stage 8E focuses on report quality, not new watermark embedding.",
        "The goal is to reduce unknown change type rate.",
        "Verification still does not access original image.",
        "The current semantic explanation remains coarse-grained unless a VLM plug-in is enabled.",
        "",
        "## Results",
        "",
        f"- total_region_reports: {summary['total_region_reports']}",
        f"- valid_report_count: {summary['valid_report_count']}",
        f"- original_unknown_rate: {summary['unknown_change_type_rate_original']}",
        f"- improved_unknown_rate: {summary['unknown_change_type_rate_improved']}",
        f"- absolute_reduction: {summary['unknown_rate_reduction_abs']}",
        f"- relative_reduction: {summary['unknown_rate_reduction_relative']}",
        f"- auth_valid_rate: {summary['auth_valid_rate']}",
        f"- capsule_valid_rate: {summary['capsule_valid_rate']}",
        f"- copyright_valid_rate: {summary['copyright_valid_rate']}",
        "",
        "No training, Stable Diffusion, SAM, CLIP, DINO, OmniGuard, 128-bit payload, or new watermark branch is used.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    stage8d_dir = (project_root / args.stage8d_dir).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    dirs = ensure_dirs(output_dir)
    failed: list[dict[str, Any]] = []
    if args.use_vlm:
        failed.append(
            {
                "failure_stage": "vlm_hook",
                "error_message": "--use_vlm is a placeholder hook; heavy VLM dependency is intentionally not implemented in Stage 8E.",
            }
        )

    region_input = read_csv(stage8d_dir / "per_region_stage8d_reports.csv")
    sample_input = read_csv(stage8d_dir / "per_sample_stage8d_summary.csv")
    stage8d_summary = json.loads((stage8d_dir / "stage8d_summary.json").read_text(encoding="utf-8"))
    selected_ids = [row["image_id"] for row in sample_input if row.get("status") == "ok"][: args.max_samples]
    selected_set = set(selected_ids)
    sample_rows = [row for row in sample_input if row.get("image_id") in selected_set]
    sample_lookup = {row["image_id"]: row for row in sample_rows}
    selected_regions = [row for row in region_input if row.get("image_id") in selected_set]

    processed: list[dict[str, Any]] = []
    for idx, row in enumerate(selected_regions, start=1):
        try:
            processed.append(process_region_row(row, sample_lookup, stage1_dir))
        except Exception as exc:
            failed.append(
                {
                    "image_id": row.get("image_id"),
                    "region_id": row.get("region_id"),
                    "failure_stage": "process_region",
                    "error_message": str(exc),
                }
            )
        if idx % 200 == 0 or idx == len(selected_regions):
            print(f"Stage 8E progress: {idx}/{len(selected_regions)} regions, failed={len(failed)}", flush=True)

    quality_summary = summarize_quality(stage8d_summary, sample_rows, processed)
    quality_summary.update(
        {
            "max_samples": args.max_samples,
            "use_vlm": bool(args.use_vlm),
            "verification_original_access": False,
            "gt_mask_usage": ["evaluation_only"],
            "robust_payload_bits": 64,
            "no_128bit_payload_attempt": True,
        }
    )
    unknown_payload = unknown_analysis(processed)
    gallery_info = build_gallery(processed, dirs["unknown_case_gallery"], args.unknown_gallery_samples)
    contact_sheet = build_contact_sheet(processed, output_dir / "stage8e_contact_sheet.png", args.contact_sheet_samples)
    quality_summary["unknown_case_gallery"] = gallery_info
    quality_summary["contact_sheet"] = contact_sheet

    write_csv(output_dir / "per_region_stage8e_reports.csv", processed, REGION_FIELDS)
    (output_dir / "stage8e_quality_summary.json").write_text(json.dumps(quality_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "stage8e_unknown_analysis.json").write_text(json.dumps(unknown_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "stage8e_failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "improved_report_examples.json").write_text(json.dumps(improved_examples(processed), ensure_ascii=False, indent=2), encoding="utf-8")
    write_overview(output_dir / "stage8e_overview.md", quality_summary)

    print(f"Saved per-region Stage 8E reports: {output_dir / 'per_region_stage8e_reports.csv'}")
    print(f"Saved quality summary: {output_dir / 'stage8e_quality_summary.json'}")
    print(f"Saved overview: {output_dir / 'stage8e_overview.md'}")
    print(json.dumps(quality_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
