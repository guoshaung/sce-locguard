from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_editguard_outputs_stage1 as stage1
import run_stage7c_non_oracle_semantic_template as s7c


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

SEVERITY_LABELS = ["background", "low", "medium", "high", "critical"]
FOREGROUND_LABELS = ["low", "medium", "high", "critical"]
METRIC_FIELDS = [
    "psnr",
    "ssim",
    "bit_accuracy",
    "ber",
    "iou",
    "dice",
    "precision",
    "recall",
    "mae",
    "region_severity_accuracy",
    "pixel_severity_accuracy",
    "per_class_accuracy_low",
    "per_class_accuracy_medium",
    "per_class_accuracy_high",
    "per_class_accuracy_critical",
    "num_pred_regions",
    "num_gt_regions",
    "fragmentation_ratio",
    "small_pred_region_ratio",
    "tiny_pred_region_ratio",
    "semantic_compactness",
]

CSV_FIELDS = [
    "variant",
    "variant_family",
    "image_id",
    "status",
    "error_message",
    "data_source",
    "original_path",
    "gt_mask_path",
    "watermarked_path",
    "tampered_path",
    "pred_mask_path",
    "prior_path",
    "prior_vis_path",
    "predicted_regions_path",
    "severity_overlay_path",
    "report_path",
    "copyright_eval_available",
    "metadata_authentication_simulated",
    "metadata_payload_bytes_options",
    "metadata_content_hash_4",
    "metadata_content_hash_8",
    "metadata_content_hash_16",
    "metadata_nonce",
    "readout_weights",
    "readout_thresholds",
    "severity_confusion_matrix",
] + METRIC_FIELDS


@dataclass(frozen=True)
class ReadoutVariant:
    name: str
    family: str


@dataclass(frozen=True)
class ReadoutConfig:
    weights: dict[str, float]
    thresholds: tuple[float, float, float]
    edge_threshold: float = 0.50


DEFAULT_REGION_CONFIG = ReadoutConfig(
    weights={"area": 0.35, "center": 0.25, "texture": 0.20, "contrast": 0.20},
    thresholds=(0.25, 0.50, 0.75),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7D non-oracle severity readout without template injection.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--input_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage7d_non_oracle_severity_readout")
    parser.add_argument("--stage7_summary", default="dfg_locguard/outputs/stage7_semantic_structured_template/stage7_summary.json")
    parser.add_argument("--stage7b_summary", default="dfg_locguard/outputs/stage7b_template_structure_optimization/stage7b_summary.json")
    parser.add_argument("--stage7c_summary", default="dfg_locguard/outputs/stage7c_non_oracle_semantic_template/stage7c_summary.json")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--recovery_threshold", type=float, default=0.2)
    parser.add_argument("--small_area_pixels", type=int, default=100)
    parser.add_argument("--tiny_area_pixels", type=int, default=25)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--calibration_samples", type=int, default=20)
    parser.add_argument("--eval_copyright", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def default_variants() -> list[ReadoutVariant]:
    return [
        ReadoutVariant("prior_mean_readout", "prior_mean_readout"),
        ReadoutVariant("region_feature_readout", "region_feature_readout"),
        ReadoutVariant("calibrated_region_feature_readout", "calibrated_region_feature_readout"),
        ReadoutVariant("hybrid_prior_region_readout", "hybrid_prior_region_readout"),
    ]


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "severity_overlays": output_dir / "severity_overlays",
        "prior_maps": output_dir / "prior_maps",
        "predicted_regions": output_dir / "predicted_regions",
        "reports": output_dir / "reports",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def variant_dir(base: Path, variant: ReadoutVariant) -> Path:
    path = base / variant.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_dirs(input_dir: Path, max_samples: int) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit())[:max_samples]


def load_rgb_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)


def load_binary_mask(path: Path, threshold: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((512, 512), NEAREST)
    return np.asarray(mask, dtype=np.uint8) > threshold


def compute_prior_bundle(original: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    height, width = gray.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    max_dist = math.sqrt(cx * cx + cy * cy)
    center = 1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(max_dist, 1e-8)
    center = np.clip(center, 0.0, 1.0).astype(np.float32)

    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = s7c.normalize_01(np.sqrt(sx * sx + sy * sy))

    mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    mean_sq = cv2.GaussianBlur(gray * gray, (0, 0), sigmaX=3.0)
    contrast = s7c.normalize_01(np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)))

    prior = 0.4 * center + 0.4 * texture + 0.2 * contrast
    prior = cv2.GaussianBlur(prior, (0, 0), sigmaX=2.0)
    prior = s7c.normalize_01(prior)
    quantized = s7c.quantize_prior(prior)
    prior_index = s7c.prior_index_map(quantized)
    meta = {
        "center_prior": "1 - normalized_distance_to_image_center",
        "texture_prior": "normalized Sobel gradient magnitude",
        "contrast_prior": "normalized local standard deviation with Gaussian sigma=3",
        "fusion": "0.4*center + 0.4*texture + 0.2*contrast, Gaussian blur sigma=2",
        "quantization": "0.00=low, 0.33=medium, 0.66=high, 1.00=critical for image-derived prior levels",
    }
    return {
        "center": center,
        "texture": texture,
        "contrast": contrast,
        "prior": prior,
        "quantized": quantized,
        "prior_index": prior_index,
        "meta": meta,
    }


def save_prior_maps(bundle: dict[str, Any], image_id: str, dirs: dict[str, Path]) -> tuple[Path, Path]:
    prior_path = dirs["prior_maps"] / f"{image_id}_c_prior.png"
    prior_vis_path = dirs["prior_maps"] / f"{image_id}_c_prior_vis.png"
    Image.fromarray(np.round(bundle["prior"] * 255.0).astype(np.uint8), mode="L").save(prior_path)
    vis = np.zeros((*bundle["prior_index"].shape, 3), dtype=np.uint8)
    for idx, label in s7c.INDEX_TO_LABEL.items():
        if idx == 0:
            continue
        vis[bundle["prior_index"] == idx] = s7c.LABEL_TO_COLOR[label]
    Image.fromarray(vis, mode="RGB").save(prior_vis_path)
    return prior_path, prior_vis_path


def label_from_score(score: float, thresholds: tuple[float, float, float]) -> str:
    low_t, med_t, high_t = thresholds
    if score < low_t:
        return "low"
    if score < med_t:
        return "medium"
    if score < high_t:
        return "high"
    return "critical"


def region_feature_score(feature: dict[str, float], config: ReadoutConfig) -> float:
    weights = config.weights
    return float(
        weights.get("area", 0.0) * feature["area_score"]
        + weights.get("center", 0.0) * feature["center_score"]
        + weights.get("texture", 0.0) * feature["texture_score"]
        + weights.get("contrast", 0.0) * feature["contrast_score"]
    )


def extract_pred_region_features(pred_mask: np.ndarray, bundle: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    height, width = pred_mask.shape
    image_area = float(height * width)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
    features: list[dict[str, Any]] = []
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
        center_score = s7c.center_score(float(cx), float(cy), width, height)
        texture_score = float(np.mean(bundle["texture"][region_mask])) if area else 0.0
        contrast_score = float(np.mean(bundle["contrast"][region_mask])) if area else 0.0
        prior_mean = float(np.mean(bundle["prior"][region_mask])) if area else 0.0
        edge_density = float(np.mean(bundle["texture"][region_mask] > DEFAULT_REGION_CONFIG.edge_threshold)) if area else 0.0
        features.append(
            {
                "region_id": int(idx),
                "bbox": [x, y, x + w, y + h],
                "area_pixels": area,
                "area_ratio": area_ratio,
                "area_score": area_score,
                "center_score": center_score,
                "texture_score": texture_score,
                "contrast_score": contrast_score,
                "edge_density": edge_density,
                "prior_mean": prior_mean,
                "centroid": [float(cx), float(cy)],
            }
        )
    return labels, features


def readout_predicted_severity(
    pred_mask: np.ndarray,
    bundle: dict[str, Any],
    variant: ReadoutVariant,
    config: ReadoutConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels, features = extract_pred_region_features(pred_mask, bundle)
    pred = np.zeros_like(pred_mask, dtype=np.uint8)
    regions: list[dict[str, Any]] = []
    for feature in features:
        region_id = int(feature["region_id"])
        if variant.family == "prior_mean_readout":
            score = float(feature["prior_mean"])
        elif variant.family in {"region_feature_readout", "calibrated_region_feature_readout"}:
            score = region_feature_score(feature, config)
        elif variant.family == "hybrid_prior_region_readout":
            score = 0.5 * float(feature["prior_mean"]) + 0.5 * region_feature_score(feature, config)
        else:
            raise ValueError(f"Unsupported readout variant: {variant.family}")
        label = label_from_score(score, config.thresholds)
        sev_idx = s7c.SEVERITY_INDEX[label]
        pred[labels == region_id] = sev_idx
        regions.append(
            {
                **feature,
                "readout_score": score,
                "predicted_severity_index": int(sev_idx),
                "predicted_severity_label": label,
            }
        )
    return pred, regions


def severity_metrics_full(
    gt_severity: np.ndarray,
    pred_severity_map: np.ndarray,
    gt_labels: np.ndarray,
    gt_regions: list[dict[str, Any]],
) -> dict[str, Any]:
    foreground = gt_severity > 0
    pixel_acc = s7c.safe_div(int((gt_severity[foreground] == pred_severity_map[foreground]).sum()), int(foreground.sum()))
    correct = 0
    total = 0
    for region in gt_regions:
        mask = gt_labels == int(region["region_id"])
        if not np.any(mask):
            continue
        total += 1
        true_idx = s7c.SEVERITY_INDEX[region["severity_label"]]
        pred_values = pred_severity_map[mask]
        values, counts = np.unique(pred_values, return_counts=True)
        pred_idx = int(values[int(np.argmax(counts))]) if values.size else 0
        if pred_idx == true_idx:
            correct += 1

    confusion = np.zeros((5, 5), dtype=np.int64)
    gt_flat = gt_severity[foreground].reshape(-1)
    pred_flat = pred_severity_map[foreground].reshape(-1)
    for gt_idx, pred_idx in zip(gt_flat, pred_flat):
        confusion[int(gt_idx), int(pred_idx)] += 1

    per_class: dict[str, float] = {}
    for label in FOREGROUND_LABELS:
        idx = s7c.SEVERITY_INDEX[label]
        per_class[f"per_class_accuracy_{label}"] = s7c.safe_div(int(confusion[idx, idx]), int(confusion[idx].sum()))

    return {
        "region_severity_accuracy": s7c.safe_div(correct, total),
        "pixel_severity_accuracy": pixel_acc,
        "severity_confusion_matrix": confusion.tolist(),
        **per_class,
    }


def metadata_hashes(image_id: str, variant: ReadoutVariant, config: ReadoutConfig, prior_meta: dict[str, str]) -> dict[str, Any]:
    nonce = hashlib.sha256(f"stage7d|{image_id}|{variant.name}".encode("utf-8")).hexdigest()[:16]
    payload = {
        "image_id": image_id,
        "variant": variant.name,
        "prior_config": prior_meta,
        "severity_thresholds": list(config.thresholds),
        "weights": config.weights,
        "nonce": nonce,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "metadata_authentication_simulated": True,
        "metadata_payload_bytes_options": "4,8,16",
        "metadata_content_hash_4": digest[:8],
        "metadata_content_hash_8": digest[:16],
        "metadata_content_hash_16": digest[:32],
        "metadata_nonce": nonce,
    }


def evaluate_copyright_if_requested(
    model: Any | None,
    tampered: np.ndarray,
    image_id: str,
    sample_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if model is None:
        return {"copyright_eval_available": False, "bit_accuracy": "", "ber": "", "copyright_error_message": "disabled"}
    try:
        seed = args.seed + int(image_id) if image_id.isdigit() else args.seed + sample_index
        message_bits = np.random.default_rng(seed).choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)
        data = stage1.image_to_editguard_batch(tampered)
        data["MES"] = None
        model.feed_data(data)
        _mask, remesg = model.image_recovery(args.recovery_threshold)
        rec = remesg.detach().float().cpu().numpy().reshape(-1)
        truth = (message_bits.reshape(-1) > 0).astype(np.uint8)
        pred = (rec > 0).astype(np.uint8)
        if pred.shape[0] != truth.shape[0]:
            raise ValueError(f"Recovered bit length mismatch: expected {truth.shape[0]}, got {pred.shape[0]}")
        bit_accuracy = float(np.mean(pred == truth))
        return {"copyright_eval_available": True, "bit_accuracy": bit_accuracy, "ber": float(1.0 - bit_accuracy)}
    except Exception as exc:
        return {"copyright_eval_available": False, "bit_accuracy": "", "ber": "", "copyright_error_message": str(exc)}


def save_overlay(
    tampered: np.ndarray,
    pred_severity_map: np.ndarray,
    predicted_regions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    overlay = s7c.severity_overlay(Image.fromarray(tampered), pred_severity_map, predicted_regions)
    overlay.save(output_path)


def error_map_image(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    out = np.zeros((*gt_bool.shape, 3), dtype=np.uint8)
    out[np.logical_and(pred_bool, gt_bool)] = [255, 255, 255]
    out[np.logical_and(pred_bool, ~gt_bool)] = [255, 80, 80]
    out[np.logical_and(~pred_bool, gt_bool)] = [80, 160, 255]
    return Image.fromarray(out, mode="RGB")


def save_contact_row(
    original: np.ndarray,
    prior_vis_path: Path,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    overlay_path: Path,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    panels = [
        s7c.title_panel(Image.fromarray(original), f"{row['variant']}"),
        s7c.title_panel(Image.open(prior_vis_path).convert("RGB"), "C_prior(I)"),
        s7c.title_panel(s7c.mask_image(gt_mask), "GT eval only"),
        s7c.title_panel(s7c.mask_image(pred_mask), f"IoU {float(row['iou']):.3f}"),
        s7c.title_panel(error_map_image(gt_mask, pred_mask), "error map"),
        s7c.title_panel(Image.open(overlay_path).convert("RGB"), f"sev {float(row['region_severity_accuracy']):.2f}"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_report(path: Path, row: dict[str, Any], prior_meta: dict[str, str], predicted_regions: list[dict[str, Any]]) -> None:
    lines = [
        f"# Stage 7D Report: {row['variant']} / {row['image_id']}",
        "",
        f"- status: {row['status']}",
        f"- data_source: {row['data_source']}",
        "- severity template injection: false",
        "- GT mask usage: deterministic tamper/evaluation/pseudo GT severity only",
        "- predicted severity readout uses: original image + EditGuard predicted mask + non-deep image priors",
        f"- metadata authentication: simulated ({row['metadata_payload_bytes_options']} bytes)",
        "",
        "## Prior",
        "",
    ]
    lines.extend([f"- {key}: {value}" for key, value in prior_meta.items()])
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- IoU: {row['iou']}",
            f"- Dice: {row['dice']}",
            f"- Precision: {row['precision']}",
            f"- Recall: {row['recall']}",
            f"- Region severity accuracy: {row['region_severity_accuracy']}",
            f"- Pixel severity accuracy: {row['pixel_severity_accuracy']}",
            f"- PSNR: {row['psnr']}",
            f"- SSIM: {row['ssim']}",
            f"- bit_accuracy: {row['bit_accuracy']}",
            f"- BER: {row['ber']}",
            "",
            "## Predicted Severity Regions",
            "",
            f"- region_count: {len(predicted_regions)}",
            f"- predicted_regions_path: {row['predicted_regions_path']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_sample_cache(
    sample_dir: Path,
    sample_index: int,
    dirs: dict[str, Path],
    args: argparse.Namespace,
    copyright_model: Any | None,
) -> dict[str, Any]:
    image_id = sample_dir.name
    original_path = sample_dir / "original.png"
    gt_mask_path = sample_dir / "gt_mask.png"
    watermarked_path = sample_dir / "watermarked.png"
    tampered_path = sample_dir / "tampered.png"
    pred_mask_path = sample_dir / "editguard_pred_mask.png"
    for required in [original_path, gt_mask_path, watermarked_path, tampered_path, pred_mask_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required Stage 1 file: {required}")

    original = load_rgb_array(original_path)
    watermarked = load_rgb_array(watermarked_path)
    tampered = load_rgb_array(tampered_path)
    gt_mask = load_binary_mask(gt_mask_path, args.mask_threshold)
    pred_mask = load_binary_mask(pred_mask_path, args.mask_threshold)
    if int(gt_mask.sum()) == 0:
        raise ValueError(f"GT mask has no foreground pixels: {gt_mask_path}")

    prior_bundle = compute_prior_bundle(original)
    prior_path, prior_vis_path = save_prior_maps(prior_bundle, image_id, dirs)
    gt_labels, gt_sev_index, gt_regions = s7c.gt_eval_severity(gt_mask)
    copyright_payload = evaluate_copyright_if_requested(copyright_model, tampered, image_id, sample_index, args)
    return {
        "image_id": image_id,
        "sample_index": sample_index,
        "sample_dir": sample_dir,
        "original_path": original_path,
        "gt_mask_path": gt_mask_path,
        "watermarked_path": watermarked_path,
        "tampered_path": tampered_path,
        "pred_mask_path": pred_mask_path,
        "original": original,
        "watermarked": watermarked,
        "tampered": tampered,
        "gt_mask": gt_mask,
        "pred_mask": pred_mask,
        "prior_bundle": prior_bundle,
        "prior_path": prior_path,
        "prior_vis_path": prior_vis_path,
        "gt_labels": gt_labels,
        "gt_sev_index": gt_sev_index,
        "gt_regions": gt_regions,
        "pixel_metrics": s7c.pixel_metrics(gt_mask, pred_mask),
        "component_metrics": s7c.component_metrics(gt_mask, pred_mask, args.small_area_pixels, args.tiny_area_pixels),
        "psnr": s7c.psnr(original, watermarked),
        "ssim": s7c.ssim_rgb(original, watermarked),
        "copyright": copyright_payload,
    }


def evaluate_readout_on_cache(
    cache: dict[str, Any],
    variant: ReadoutVariant,
    config: ReadoutConfig,
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path | None]:
    image_id = cache["image_id"]
    variant_dir(dirs["severity_overlays"], variant)
    variant_dir(dirs["predicted_regions"], variant)

    paths = {
        "predicted_regions": dirs["predicted_regions"] / variant.name / f"{image_id}_predicted_regions.json",
        "severity_overlay": dirs["severity_overlays"] / variant.name / f"{image_id}_severity_overlay.png",
        "contact_row": dirs["severity_overlays"] / variant.name / f"{image_id}_contact_row.png",
        "report": dirs["reports"] / f"{variant.name}_{image_id}_stage7d_report.md",
        "metrics": dirs["reports"] / f"{variant.name}_{image_id}_metrics.json",
    }
    if paths["metrics"].exists() and not args.overwrite:
        row = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        return row, paths["contact_row"] if paths["contact_row"].exists() else None

    pred_sev_map, predicted_regions = readout_predicted_severity(
        cache["pred_mask"], cache["prior_bundle"], variant, config
    )
    paths["predicted_regions"].write_text(json.dumps(predicted_regions, ensure_ascii=False, indent=2), encoding="utf-8")
    save_overlay(cache["tampered"], pred_sev_map, predicted_regions, paths["severity_overlay"])

    sev = severity_metrics_full(cache["gt_sev_index"], pred_sev_map, cache["gt_labels"], cache["gt_regions"])
    metadata = metadata_hashes(image_id, variant, config, cache["prior_bundle"]["meta"])
    copyright_payload = cache["copyright"]
    row: dict[str, Any] = {
        "variant": variant.name,
        "variant_family": variant.family,
        "image_id": image_id,
        "status": "ok",
        "error_message": "",
        "data_source": "stage1_editguard_stage1_generated_existing_pred_mask",
        "original_path": str(cache["original_path"].resolve()),
        "gt_mask_path": str(cache["gt_mask_path"].resolve()),
        "watermarked_path": str(cache["watermarked_path"].resolve()),
        "tampered_path": str(cache["tampered_path"].resolve()),
        "pred_mask_path": str(cache["pred_mask_path"].resolve()),
        "prior_path": str(cache["prior_path"].resolve()),
        "prior_vis_path": str(cache["prior_vis_path"].resolve()),
        "predicted_regions_path": str(paths["predicted_regions"].resolve()),
        "severity_overlay_path": str(paths["severity_overlay"].resolve()),
        "report_path": str(paths["report"].resolve()),
        "copyright_eval_available": copyright_payload["copyright_eval_available"],
        "bit_accuracy": copyright_payload["bit_accuracy"],
        "ber": copyright_payload["ber"],
        "metadata_authentication_simulated": metadata["metadata_authentication_simulated"],
        "metadata_payload_bytes_options": metadata["metadata_payload_bytes_options"],
        "metadata_content_hash_4": metadata["metadata_content_hash_4"],
        "metadata_content_hash_8": metadata["metadata_content_hash_8"],
        "metadata_content_hash_16": metadata["metadata_content_hash_16"],
        "metadata_nonce": metadata["metadata_nonce"],
        "readout_weights": json.dumps(config.weights, sort_keys=True),
        "readout_thresholds": json.dumps(list(config.thresholds)),
        "severity_confusion_matrix": json.dumps(sev["severity_confusion_matrix"]),
        "psnr": cache["psnr"],
        "ssim": cache["ssim"],
    }
    row.update(cache["pixel_metrics"])
    row.update(cache["component_metrics"])
    for key, value in sev.items():
        if key != "severity_confusion_matrix":
            row[key] = value

    if args.save_visualizations:
        save_contact_row(
            cache["original"],
            cache["prior_vis_path"],
            cache["gt_mask"],
            cache["pred_mask"],
            paths["severity_overlay"],
            row,
            paths["contact_row"],
        )
    write_report(paths["report"], row, cache["prior_bundle"]["meta"], predicted_regions)
    paths["metrics"].write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row, paths["contact_row"] if paths["contact_row"].exists() else None


def candidate_configs() -> list[ReadoutConfig]:
    weights = [
        {"area": 0.35, "center": 0.25, "texture": 0.20, "contrast": 0.20},
        {"area": 0.25, "center": 0.35, "texture": 0.20, "contrast": 0.20},
        {"area": 0.25, "center": 0.25, "texture": 0.25, "contrast": 0.25},
        {"area": 0.45, "center": 0.20, "texture": 0.20, "contrast": 0.15},
        {"area": 0.20, "center": 0.30, "texture": 0.25, "contrast": 0.25},
        {"area": 0.30, "center": 0.20, "texture": 0.25, "contrast": 0.25},
    ]
    thresholds = [
        (0.25, 0.50, 0.75),
        (0.20, 0.45, 0.70),
        (0.15, 0.40, 0.65),
        (0.30, 0.55, 0.78),
        (0.35, 0.60, 0.82),
    ]
    return [ReadoutConfig(w, t) for w in weights for t in thresholds]


def calibrate_region_readout(caches: list[dict[str, Any]], max_samples: int) -> dict[str, Any]:
    variant = ReadoutVariant("calibrated_region_feature_readout", "calibrated_region_feature_readout")
    calibration_caches = caches[: min(max_samples, len(caches))]
    candidates = candidate_configs()
    best_payload: dict[str, Any] | None = None
    for idx, config in enumerate(candidates):
        region_accs: list[float] = []
        pixel_accs: list[float] = []
        for cache in calibration_caches:
            pred_sev_map, _ = readout_predicted_severity(cache["pred_mask"], cache["prior_bundle"], variant, config)
            sev = severity_metrics_full(cache["gt_sev_index"], pred_sev_map, cache["gt_labels"], cache["gt_regions"])
            region_accs.append(float(sev["region_severity_accuracy"]))
            pixel_accs.append(float(sev["pixel_severity_accuracy"]))
        payload = {
            "candidate_index": idx,
            "weights": config.weights,
            "thresholds": list(config.thresholds),
            "calibration_samples": len(calibration_caches),
            "mean_region_severity_accuracy": float(np.mean(region_accs)) if region_accs else 0.0,
            "mean_pixel_severity_accuracy": float(np.mean(pixel_accs)) if pixel_accs else 0.0,
        }
        if best_payload is None or (
            payload["mean_region_severity_accuracy"],
            payload["mean_pixel_severity_accuracy"],
        ) > (
            best_payload["mean_region_severity_accuracy"],
            best_payload["mean_pixel_severity_accuracy"],
        ):
            best_payload = payload
    assert best_payload is not None
    best_payload["note"] = "Grid calibration only; no neural network training or template injection."
    return best_payload


def config_for_variant(variant: ReadoutVariant, calibration_payload: dict[str, Any]) -> ReadoutConfig:
    if variant.family == "calibrated_region_feature_readout":
        return ReadoutConfig(
            weights={key: float(value) for key, value in calibration_payload["weights"].items()},
            thresholds=tuple(float(value) for value in calibration_payload["thresholds"]),  # type: ignore[arg-type]
        )
    return DEFAULT_REGION_CONFIG


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def mean(values: list[Any]) -> float:
    vals: list[float] = []
    for value in values:
        if value in ("", None):
            continue
        try:
            vals.append(float(value))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else 0.0


def aggregate_confusion(rows: list[dict[str, Any]]) -> list[list[int]]:
    confusion = np.zeros((5, 5), dtype=np.int64)
    for row in rows:
        try:
            confusion += np.asarray(json.loads(row.get("severity_confusion_matrix", "[]")), dtype=np.int64)
        except Exception:
            continue
    return confusion.tolist()


def summarize_by_variant(rows: list[dict[str, Any]], variants: list[ReadoutVariant]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in variants:
        group_rows = [row for row in rows if row.get("variant") == variant.name and row.get("status") == "ok"]
        out[variant.name] = {
            "variant_family": variant.family,
            "evaluated_samples": len(group_rows),
            "mean_metrics": {field: mean([row.get(field) for row in group_rows]) for field in METRIC_FIELDS},
            "severity_confusion_matrix_labels": SEVERITY_LABELS,
            "severity_confusion_matrix": aggregate_confusion(group_rows),
        }
    return out


def build_contact_sheet(vis_paths: list[Path], output_path: Path, max_items: int) -> dict[str, Any]:
    existing = [path for path in vis_paths if path and path.exists()]
    if len(existing) > max_items:
        indices = sorted({round(i * (len(existing) - 1) / (max_items - 1)) for i in range(max_items)})
        selected = [existing[index] for index in indices]
        sampling = "even"
    else:
        selected = existing
        sampling = "all"
    rows = [Image.open(path).convert("RGB").resize((960, 165), BICUBIC) for path in selected]
    if not rows:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}
    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(rows)}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_stage7b(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    return s7c.best_stage7b(summary)


def pick_best(summary_by_variant: dict[str, Any]) -> dict[str, str | None]:
    names = list(summary_by_variant)
    if not names:
        return {"best_localization": None, "best_severity": None, "best_balanced": None}

    def metric(name: str, field: str) -> float:
        return float(summary_by_variant[name]["mean_metrics"].get(field, 0.0))

    best_loc = max(names, key=lambda name: (metric(name, "iou"), metric(name, "dice")))
    best_sev = max(names, key=lambda name: (metric(name, "region_severity_accuracy"), metric(name, "pixel_severity_accuracy")))
    best_bal = max(
        names,
        key=lambda name: (
            0.40 * metric(name, "iou")
            + 0.20 * metric(name, "dice")
            + 0.25 * metric(name, "region_severity_accuracy")
            + 0.15 * metric(name, "pixel_severity_accuracy")
        ),
    )
    return {"best_localization": best_loc, "best_severity": best_sev, "best_balanced": best_bal}


def write_overview(
    output_path: Path,
    stage7_metrics: dict[str, Any] | None,
    stage7b_best: dict[str, Any] | None,
    stage7c_base: dict[str, Any] | None,
    summary_by_variant: dict[str, Any],
    best: dict[str, str | None],
) -> None:
    lines = [
        "# Stage 7D Non-Oracle Severity Readout without Template Injection",
        "",
        "Stage 7C base_only_control is used as the stable localization backbone.",
        "Stage 7D does not inject severity into the fragile template.",
        "Stage 7D interprets severity after localization using non-oracle image-derived priors and predicted tamper regions.",
        "This avoids localization degradation caused by template-level prior injection.",
        "",
        "GT masks are used only for evaluation and pseudo ground-truth severity labels. They are not used for predicted severity readout.",
        "The metadata authentication fields are simulated low-capacity hashes; RSA-PSS is not implemented in this stage.",
        "",
        "## References",
        "",
    ]
    if stage7_metrics:
        lines.append(
            f"- Stage 7 pure 5-level template: IoU={stage7_metrics.get('iou')}, Dice={stage7_metrics.get('dice')}, "
            f"region_severity_accuracy={stage7_metrics.get('region_severity_accuracy')}, fragmentation_ratio={stage7_metrics.get('fragmentation_ratio')}"
        )
    if stage7b_best:
        m = stage7b_best["mean_metrics"]
        lines.append(
            f"- Stage 7B oracle upper bound ({stage7b_best['name']}): IoU={m.get('iou')}, Dice={m.get('dice')}, "
            f"region_severity_accuracy={m.get('region_severity_accuracy')}, fragmentation_ratio={m.get('fragmentation_ratio')}"
        )
    if stage7c_base:
        m = stage7c_base.get("mean_metrics", {})
        lines.append(
            f"- Stage 7C base_only_control: IoU={m.get('iou')}, Dice={m.get('dice')}, "
            f"region_severity_accuracy={m.get('region_severity_accuracy')}, fragmentation_ratio={m.get('fragmentation_ratio')}"
        )

    lines.extend(["", "## Stage 7D Variants", ""])
    lines.append("| Variant | IoU | Dice | Region Sev Acc | Pixel Sev Acc | Frag Ratio | PSNR | SSIM | Bit Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, payload in summary_by_variant.items():
        m = payload["mean_metrics"]
        lines.append(
            f"| {name} | {m['iou']:.6f} | {m['dice']:.6f} | {m['region_severity_accuracy']:.6f} | "
            f"{m['pixel_severity_accuracy']:.6f} | {m['fragmentation_ratio']:.6f} | {m['psnr']:.6f} | "
            f"{m['ssim']:.6f} | {m['bit_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Best localization: {best.get('best_localization')}",
            f"- Best severity readout: {best.get('best_severity')}",
            f"- Best balanced Stage 7D candidate: {best.get('best_balanced')}",
            "- No training, Stable Diffusion, SAM, CLIP, DINO, OmniGuard, third watermark channel, or robust payload increase is used.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    dirs = ensure_dirs(output_dir)
    variants = default_variants()
    samples = sample_dirs(input_dir, args.max_samples)
    if not samples:
        raise FileNotFoundError(f"No Stage 1 generated samples found under {input_dir}")

    copyright_model = None
    if args.eval_copyright:
        copyright_model = stage1.load_editguard_model(project_root, project_root / args.opt, project_root / args.ckpt)

    caches: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for sample_index, sample_dir in enumerate(samples):
        try:
            caches.append(load_sample_cache(sample_dir, sample_index, dirs, args, copyright_model))
        except Exception as exc:
            failed.append(
                {
                    "variant": "sample_cache",
                    "image_id": sample_dir.name,
                    "failure_stage": "load_sample_cache",
                    "error_message": str(exc),
                }
            )
        if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(samples):
            print(f"Stage 7D cache progress: {sample_index + 1}/{len(samples)} samples, failed={len(failed)}", flush=True)

    calibration_payload = calibrate_region_readout(caches, args.calibration_samples)
    (output_dir / "stage7d_calibration_config.json").write_text(
        json.dumps(calibration_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    vis_paths: list[Path] = []
    total_cases = len(caches) * len(variants)
    done = 0
    for variant in variants:
        config = config_for_variant(variant, calibration_payload)
        for cache in caches:
            done += 1
            try:
                row, vis_path = evaluate_readout_on_cache(cache, variant, config, dirs, args)
                rows.append(row)
                if vis_path is not None:
                    vis_paths.append(vis_path)
            except Exception as exc:
                failure = {
                    "variant": variant.name,
                    "image_id": cache.get("image_id", ""),
                    "failure_stage": "evaluate_readout",
                    "error_message": str(exc),
                }
                failed.append(failure)
                rows.append(
                    {
                        "variant": variant.name,
                        "variant_family": variant.family,
                        "image_id": cache.get("image_id", ""),
                        "status": "failed",
                        "error_message": str(exc),
                    }
                )
            if done % 25 == 0 or done == total_cases:
                print(f"Stage 7D readout progress: {done}/{total_cases} cases, failed={len(failed)}", flush=True)

    contact_sheet = build_contact_sheet(vis_paths, output_dir / "stage7d_contact_sheet.png", args.contact_sheet_samples)
    write_csv(output_dir / "per_sample_stage7d_metrics.csv", rows)
    (output_dir / "stage7d_failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_by_variant = summarize_by_variant(rows, variants)
    best = pick_best(summary_by_variant)
    stage7_summary = load_json((project_root / args.stage7_summary).resolve())
    stage7_metrics = stage7_summary.get("mean_metrics") if stage7_summary else None
    stage7b_best = best_stage7b(load_json((project_root / args.stage7b_summary).resolve()))
    stage7c_summary = load_json((project_root / args.stage7c_summary).resolve())
    stage7c_base = stage7c_summary.get("stage7c_base_only_control") if stage7c_summary else None
    summary = {
        "stage": "stage7d_non_oracle_severity_readout",
        "total_cases": len(rows),
        "evaluated_cases": len([row for row in rows if row.get("status") == "ok"]),
        "failed_cases": len(failed),
        "max_samples": args.max_samples,
        "data_source": "dfg_locguard/outputs/editguard_stage1_generated existing original/watermarked/tampered/editguard_pred_mask",
        "localization_backbone": "Stage 1 EditGuard/base_only predicted mask",
        "template_severity_injection": False,
        "gt_mask_usage": ["evaluation", "pseudo_ground_truth_severity_labels"],
        "gt_mask_used_for_predicted_severity_readout": False,
        "metadata_authentication": {
            "simulated": True,
            "metadata_payload_bytes": [4, 8, 16],
            "content": "hash(image_id || prior_config || severity_thresholds || nonce)",
            "rsa_pss_implemented": False,
        },
        "calibration": calibration_payload,
        "variants": summary_by_variant,
        "stage7_pure_5_level_reference": stage7_metrics,
        "stage7b_oracle_best_reference": stage7b_best,
        "stage7c_base_only_control_reference": stage7c_base,
        "best_variants": best,
        "contact_sheet": contact_sheet,
        "constraints": {
            "no_training": True,
            "no_stable_diffusion": True,
            "no_sam_clip_dino": True,
            "no_omniguard_switch": True,
            "third_watermark_channel": False,
            "robust_payload_increase": 0,
            "previous_stage_outputs_modified": False,
        },
    }
    (output_dir / "stage7d_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_overview(output_dir / "stage7d_overview.md", stage7_metrics, stage7b_best, stage7c_base, summary_by_variant, best)

    if copyright_model is not None:
        del copyright_model
        torch.cuda.empty_cache()

    print(f"Saved metrics: {output_dir / 'per_sample_stage7d_metrics.csv'}")
    print(f"Saved summary: {output_dir / 'stage7d_summary.json'}")
    print(f"Saved overview: {output_dir / 'stage7d_overview.md'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
