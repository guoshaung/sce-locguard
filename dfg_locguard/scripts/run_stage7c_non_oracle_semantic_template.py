from __future__ import annotations

import argparse
import csv
import gc
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


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

SEVERITY_LABELS = ["background", "low", "medium", "high", "critical"]
SEVERITY_INDEX = {"background": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
INDEX_TO_LABEL = {idx: label for label, idx in SEVERITY_INDEX.items()}
PRIOR_CODE = {"low": 0.0, "medium": 0.33, "high": 0.66, "critical": 1.0}
LABEL_TO_COLOR = {
    "background": (0, 0, 0),
    "low": (80, 180, 120),
    "medium": (245, 190, 70),
    "high": (235, 120, 55),
    "critical": (220, 55, 80),
}

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
    "epsilon",
    "image_id",
    "status",
    "error_message",
    "base_template_source",
    "normalization",
    "original_path",
    "gt_mask_path",
    "prior_path",
    "prior_vis_path",
    "hybrid_template_path",
    "hybrid_template_vis_path",
    "semantic_watermarked_path",
    "semantic_tampered_path",
    "recovered_hybrid_template_path",
    "semantic_pred_mask_path",
    "predicted_severity_regions_path",
    "severity_overlay_path",
    "report_path",
    "copyright_eval_available",
    "semantic_template_levels",
    "extra_watermark_channel",
    "robust_payload_increase",
] + METRIC_FIELDS


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    epsilon: float
    coarse_size: int | None = None
    normalization: str = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7C non-oracle semantic template validation.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--input_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage7c_non_oracle_semantic_template")
    parser.add_argument("--stage7_summary", default="dfg_locguard/outputs/stage7_semantic_structured_template/stage7_summary.json")
    parser.add_argument("--stage7b_summary", default="dfg_locguard/outputs/stage7b_template_structure_optimization/stage7b_summary.json")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--small_area_pixels", type=int, default=100)
    parser.add_argument("--tiny_area_pixels", type=int, default=25)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_visualizations", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def default_variants() -> list[Variant]:
    return [
        Variant("base_only_control", "base_only_control", 0.0, normalization="original_editguard_blue_secret_full_support"),
        Variant(
            "nonoracle_coarse_prior_eps0p05",
            "nonoracle_coarse_prior",
            0.05,
            coarse_size=32,
            normalization="rgb_blue_backbone_plus_upsampled_prior_divided_by_1_plus_epsilon",
        ),
        Variant(
            "nonoracle_coarse_prior_eps0p10",
            "nonoracle_coarse_prior",
            0.10,
            coarse_size=32,
            normalization="rgb_blue_backbone_plus_upsampled_prior_divided_by_1_plus_epsilon",
        ),
        Variant(
            "nonoracle_offset_prior_eps0p05",
            "nonoracle_offset_prior",
            0.05,
            normalization="rgb_blue_backbone_plus_full_resolution_prior_divided_by_1_plus_epsilon",
        ),
    ]


def configure_editguard_imports(project_root: Path) -> None:
    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def load_editguard_model(project_root: Path, opt_path: Path, ckpt_path: Path):
    configure_editguard_imports(project_root)
    import options.options as option
    from models import create_model as create_model_editguard

    opt = option.parse(str(opt_path), is_train=False)
    opt["sdinpaint"] = False
    opt["controlnetinpaint"] = False
    opt["sdxl"] = False
    opt["repaint"] = False
    opt["dist"] = False
    opt = option.dict_to_nonedict(opt)
    torch.backends.cudnn.benchmark = True
    model = create_model_editguard(opt)
    model.load_test(str(ckpt_path))
    return model


def sample_dirs(input_dir: Path, max_samples: int) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit())[:max_samples]


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "priors": output_dir / "priors",
        "hybrid_templates": output_dir / "hybrid_templates",
        "recovered_templates": output_dir / "recovered_templates",
        "predicted_masks": output_dir / "predicted_masks",
        "severity_overlays": output_dir / "severity_overlays",
        "semantic_watermarked": output_dir / "semantic_watermarked",
        "semantic_tampered": output_dir / "semantic_tampered",
        "reports": output_dir / "reports",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def variant_path(base: Path, variant: Variant) -> Path:
    path = base / variant.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((512, 512), BICUBIC)


def load_mask(path: Path, threshold: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((512, 512), NEAREST)
    return np.asarray(mask, dtype=np.uint8) > threshold


def normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_image_prior(original: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    max_dist = math.sqrt(cx * cx + cy * cy)
    center = 1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(max_dist, 1e-8)
    center = np.clip(center, 0.0, 1.0).astype(np.float32)

    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = normalize_01(np.sqrt(sx * sx + sy * sy))

    mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    mean_sq = cv2.GaussianBlur(gray * gray, (0, 0), sigmaX=3.0)
    contrast = normalize_01(np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)))

    prior = 0.4 * center + 0.4 * texture + 0.2 * contrast
    prior = cv2.GaussianBlur(prior, (0, 0), sigmaX=2.0)
    prior = normalize_01(prior)
    quantized = quantize_prior(prior)
    meta = {
        "center_prior": "1 - normalized_distance_to_image_center",
        "texture_prior": "normalized Sobel gradient magnitude",
        "contrast_prior": "normalized local standard deviation with Gaussian sigma=3",
        "fusion": "0.4*center + 0.4*texture + 0.2*contrast, Gaussian blur sigma=2, quantized to 0/0.33/0.66/1.0",
    }
    return prior, quantized, meta


def quantize_prior(prior: np.ndarray) -> np.ndarray:
    q = np.zeros_like(prior, dtype=np.float32)
    q[(prior >= 0.25) & (prior < 0.50)] = 0.33
    q[(prior >= 0.50) & (prior < 0.75)] = 0.66
    q[prior >= 0.75] = 1.0
    return q


def prior_index_map(quantized: np.ndarray) -> np.ndarray:
    idx = np.ones_like(quantized, dtype=np.uint8)
    idx[(quantized >= 0.25) & (quantized < 0.50)] = SEVERITY_INDEX["medium"]
    idx[(quantized >= 0.50) & (quantized < 0.83)] = SEVERITY_INDEX["high"]
    idx[quantized >= 0.83] = SEVERITY_INDEX["critical"]
    return idx


def original_editguard_base_template(height: int = 512, width: int = 512) -> np.ndarray:
    # Stage 1's current EditGuard localization secret is full-support blue: RGB [0, 0, 1].
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 2] = 1.0
    return base


def build_hybrid_template(base: np.ndarray, prior_quantized: np.ndarray, variant: Variant) -> tuple[np.ndarray, np.ndarray, str]:
    if variant.family == "base_only_control":
        return base.copy(), np.zeros_like(prior_quantized), variant.normalization
    prior_payload = prior_quantized.astype(np.float32)
    if variant.family == "nonoracle_coarse_prior":
        if variant.coarse_size is None:
            raise ValueError("coarse_size is required for nonoracle_coarse_prior")
        low = cv2.resize(prior_payload, (variant.coarse_size, variant.coarse_size), interpolation=cv2.INTER_NEAREST)
        prior_payload = cv2.resize(low, prior_payload.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.float32)
    elif variant.family != "nonoracle_offset_prior":
        raise ValueError(f"Unsupported Stage 7C variant family: {variant.family}")
    prior_rgb = np.repeat(prior_payload[:, :, None], 3, axis=2)
    raw = base + float(variant.epsilon) * prior_rgb
    template = np.clip(raw / (1.0 + float(variant.epsilon)), 0.0, 1.0).astype(np.float32)
    return template, prior_payload, variant.normalization


def save_prior(prior: np.ndarray, quantized: np.ndarray, image_id: str, dirs: dict[str, Path]) -> tuple[Path, Path]:
    prior_path = dirs["priors"] / f"{image_id}_c_prior.png"
    prior_vis_path = dirs["priors"] / f"{image_id}_c_prior_vis.png"
    Image.fromarray(np.round(prior * 255.0).astype(np.uint8), mode="L").save(prior_path)
    vis = np.zeros((*quantized.shape, 3), dtype=np.uint8)
    idx = prior_index_map(quantized)
    for label_idx, label in INDEX_TO_LABEL.items():
        if label_idx == 0:
            continue
        vis[idx == label_idx] = LABEL_TO_COLOR[label]
    Image.fromarray(vis, mode="RGB").save(prior_vis_path)
    return prior_path, prior_vis_path


def save_template(template: np.ndarray, template_path: Path, vis_path: Path) -> None:
    Image.fromarray(np.round(template * 255.0).astype(np.uint8), mode="RGB").save(template_path)
    gray = template.mean(axis=2)
    Image.fromarray(np.round(gray * 255.0).astype(np.uint8), mode="L").convert("RGB").save(vis_path)


def image_to_editguard_batch(image: np.ndarray, secret_template: np.ndarray) -> dict[str, torch.Tensor]:
    img_gt = image.astype(np.float32) / 255.0
    img_gt = img_gt[:, :, [2, 1, 0]]
    img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float().unsqueeze(0)
    img_gt = torch.nn.functional.interpolate(img_gt, size=(512, 512), mode="nearest", align_corners=None)
    img_gt = img_gt.unsqueeze(0)

    secret = np.expand_dims(secret_template.astype(np.float32), axis=0)
    imgs_lq = torch.from_numpy(np.ascontiguousarray(secret)).float().permute(0, 3, 1, 2).unsqueeze(0)
    list_h = torch.stack([imgs_lq], dim=0)
    return {"LQ": list_h, "GT": img_gt}


def message_bits_for(image_id: str, seed: int, sample_index: int) -> np.ndarray:
    rng_seed = seed + int(image_id) if image_id.isdigit() else seed + sample_index
    rng = np.random.default_rng(rng_seed)
    return rng.choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)


def deterministic_tamper(watermarked: np.ndarray, mask: np.ndarray, sample_index: int) -> np.ndarray:
    tampered = watermarked.copy()
    colors = np.asarray(
        [
            [35, 145, 235],
            [235, 90, 60],
            [80, 190, 120],
            [190, 80, 220],
            [245, 185, 45],
        ],
        dtype=np.uint8,
    )
    replacement = np.zeros_like(tampered)
    replacement[:] = colors[sample_index % len(colors)]
    tampered[mask] = (0.25 * tampered[mask] + 0.75 * replacement[mask]).astype(np.uint8)
    return tampered


def recover_template(model: Any, tampered: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    from models.IBSN import iwt

    data = image_to_editguard_batch(tampered, template)
    data["MES"] = None
    model.feed_data(data)
    model.netG.eval()
    with torch.no_grad():
        b, t, c, h, w = model.real_H.shape
        center = t // 2
        intval = model.gop // 2
        host = model.real_H[:, center - intval : center + intval + 1]
        y = model.Quantization(host.squeeze(1))
        _out_x, out_x_h, _out_z, recmessage = model.netG(x=y, rev=True)
        out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
        out_x_h = torch.stack(out_x_h, dim=1)
        out_x_h = out_x_h.reshape(-1, 1, model.gop, 3, h, w)
        rec_loc = out_x_h[:, :, model.gop // 2]
        rec_rgb = rec_loc[0, 0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return rec_rgb.astype(np.float32), recmessage


def evaluate_copyright(message_bits: np.ndarray, recmessage: torch.Tensor) -> dict[str, Any]:
    try:
        rec = recmessage.detach().float().cpu().numpy().reshape(-1)
        truth = (message_bits.reshape(-1) > 0).astype(np.uint8)
        pred = (rec > 0).astype(np.uint8)
        if pred.shape[0] != truth.shape[0]:
            raise ValueError(f"Recovered bit length mismatch: expected {truth.shape[0]}, got {pred.shape[0]}")
        bit_accuracy = float(np.mean(pred == truth))
        return {"copyright_eval_available": True, "bit_accuracy": bit_accuracy, "ber": float(1.0 - bit_accuracy)}
    except Exception as exc:
        return {"copyright_eval_available": False, "bit_accuracy": "", "ber": "", "copyright_error_message": str(exc)}


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def pixel_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())
    total = tp + fp + fn + tn
    return {
        "iou": safe_div(tp, tp + fp + fn),
        "dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "mae": safe_div(fp + fn, total),
    }


def component_metrics(gt: np.ndarray, pred: np.ndarray, small_area_pixels: int, tiny_area_pixels: int) -> dict[str, Any]:
    num_pred, _pred_labels, pred_stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    num_gt, _gt_labels, _gt_stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
    pred_region_count = int(max(num_pred - 1, 0))
    gt_region_count = int(max(num_gt - 1, 0))
    areas = [int(pred_stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, num_pred)]
    total_area = int(sum(areas))
    largest_area = int(max(areas, default=0))
    small_count = int(sum(area < small_area_pixels for area in areas))
    tiny_count = int(sum(area < tiny_area_pixels for area in areas))
    return {
        "num_pred_regions": pred_region_count,
        "num_gt_regions": gt_region_count,
        "fragmentation_ratio": safe_div(pred_region_count, max(gt_region_count, 1)),
        "small_pred_region_ratio": safe_div(small_count, max(pred_region_count, 1)),
        "tiny_pred_region_ratio": safe_div(tiny_count, max(pred_region_count, 1)),
        "semantic_compactness": safe_div(largest_area, total_area),
    }


def psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    diff = original.astype(np.float64) - watermarked.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def ssim_rgb(original: np.ndarray, watermarked: np.ndarray) -> float:
    return float(np.mean([ssim_channel(original[:, :, ch], watermarked[:, :, ch]) for ch in range(3)]))


def center_score(cx: float, cy: float, width: int, height: int) -> float:
    image_cx = (width - 1) / 2.0
    image_cy = (height - 1) / 2.0
    max_dist = math.sqrt(image_cx * image_cx + image_cy * image_cy)
    dist = math.sqrt((cx - image_cx) ** 2 + (cy - image_cy) ** 2)
    return float(np.clip(1.0 - safe_div(dist, max_dist), 0.0, 1.0))


def severity_from_score(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "medium"
    if score < 0.75:
        return "high"
    return "critical"


def gt_eval_severity(gt_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    height, width = gt_mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_mask.astype(np.uint8), connectivity=8)
    severity_index = np.zeros((height, width), dtype=np.uint8)
    regions: list[dict[str, Any]] = []
    image_area = float(height * width)
    for idx in range(1, num_labels):
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
        score = float(0.7 * area_score + 0.3 * c_score)
        label = severity_from_score(score)
        severity_index[labels == idx] = SEVERITY_INDEX[label]
        regions.append(
            {
                "region_id": int(idx),
                "bbox": [x, y, x + w, y + h],
                "area_pixels": area,
                "area_ratio": area_ratio,
                "severity_score": score,
                "severity_label": label,
            }
        )
    return labels, severity_index, regions


def predicted_severity(pred_mask: np.ndarray, prior_index: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
    pred = np.zeros_like(prior_index, dtype=np.uint8)
    regions: list[dict[str, Any]] = []
    image_area = float(pred_mask.size)
    for idx in range(1, num_labels):
        region_mask = labels == idx
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        values, counts = np.unique(prior_index[region_mask], return_counts=True)
        values = values[values > 0]
        if values.size:
            count_lookup = {int(v): int((prior_index[region_mask] == v).sum()) for v in values}
            pred_idx = max(count_lookup, key=count_lookup.get)
        else:
            pred_idx = SEVERITY_INDEX["low"]
        pred[region_mask] = pred_idx
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[idx]
        regions.append(
            {
                "region_id": int(idx),
                "bbox": [x, y, x + w, y + h],
                "area_pixels": area,
                "area_ratio": float(area / image_area),
                "centroid": [float(cx), float(cy)],
                "predicted_severity_index": int(pred_idx),
                "predicted_severity_label": INDEX_TO_LABEL[int(pred_idx)],
            }
        )
    return pred, regions


def severity_metrics(gt_severity: np.ndarray, pred_severity_map: np.ndarray, gt_labels: np.ndarray, gt_regions: list[dict[str, Any]]) -> dict[str, Any]:
    foreground = gt_severity > 0
    pixel_acc = safe_div(int((gt_severity[foreground] == pred_severity_map[foreground]).sum()), int(foreground.sum()))
    correct = 0
    total = 0
    for region in gt_regions:
        mask = gt_labels == int(region["region_id"])
        if not np.any(mask):
            continue
        total += 1
        true_idx = SEVERITY_INDEX[region["severity_label"]]
        pred_values = pred_severity_map[mask]
        values, counts = np.unique(pred_values, return_counts=True)
        pred_idx = int(values[int(np.argmax(counts))]) if values.size else 0
        if pred_idx == true_idx:
            correct += 1
    return {
        "region_severity_accuracy": safe_div(correct, total),
        "pixel_severity_accuracy": pixel_acc,
    }


def severity_color_image(index_map: np.ndarray) -> Image.Image:
    vis = np.zeros((*index_map.shape, 3), dtype=np.uint8)
    for idx, label in INDEX_TO_LABEL.items():
        vis[index_map == idx] = LABEL_TO_COLOR[label]
    return Image.fromarray(vis, mode="RGB")


def severity_overlay(tampered: Image.Image, pred_severity_map: np.ndarray, regions: list[dict[str, Any]]) -> Image.Image:
    base = np.asarray(tampered.convert("RGB"), dtype=np.float32)
    color = np.asarray(severity_color_image(pred_severity_map), dtype=np.float32)
    active = pred_severity_map > 0
    overlay = base.copy()
    overlay[active] = 0.55 * base[active] + 0.45 * color[active]
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for region in regions[:30]:
        x1, y1, x2, y2 = region["bbox"]
        label = region["predicted_severity_label"]
        draw.rectangle([x1, y1, x2, y2], outline=LABEL_TO_COLOR[label], width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=LABEL_TO_COLOR[label])
    return image


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (160, 160)) -> Image.Image:
    header_h = 28
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title[:24], fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def save_contact_row(
    original: Image.Image,
    prior_vis: Image.Image,
    template_vis: Image.Image,
    watermarked: Image.Image,
    recovered: Image.Image,
    pred_mask: np.ndarray,
    overlay: Image.Image,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    panels = [
        title_panel(original, f"{row['variant']}"),
        title_panel(prior_vis, "C_prior(I)"),
        title_panel(template_vis, "S_hybrid"),
        title_panel(watermarked, f"PSNR {float(row['psnr']):.2f}"),
        title_panel(recovered, "recovered"),
        title_panel(mask_image(pred_mask), f"IoU {float(row['iou']):.3f}"),
        title_panel(overlay, f"sev {float(row['region_severity_accuracy']):.2f}"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(output_path)


def write_report(path: Path, row: dict[str, Any], prior_meta: dict[str, str]) -> None:
    lines = [
        f"# Stage 7C Report: {row['variant']} / {row['image_id']}",
        "",
        f"- status: {row['status']}",
        f"- failure reason: {row.get('error_message') or 'none'}",
        f"- base_template_source: {row['base_template_source']}",
        f"- normalization: {row['normalization']}",
        f"- GT mask usage: deterministic tamper and evaluation only",
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
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_case(
    model: Any,
    sample_dir: Path,
    sample_index: int,
    variant: Variant,
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path | None]:
    image_id = sample_dir.name
    original_path = sample_dir / "original.png"
    gt_mask_path = sample_dir / "gt_mask.png"
    if not original_path.exists() or not gt_mask_path.exists():
        raise FileNotFoundError(f"Missing original or gt_mask in {sample_dir}")

    for key in ["hybrid_templates", "recovered_templates", "predicted_masks", "severity_overlays", "semantic_watermarked", "semantic_tampered"]:
        variant_path(dirs[key], variant)

    paths = {
        "prior": dirs["priors"] / f"{image_id}_c_prior.png",
        "prior_vis": dirs["priors"] / f"{image_id}_c_prior_vis.png",
        "hybrid_template": dirs["hybrid_templates"] / variant.name / f"{image_id}_hybrid_template.png",
        "hybrid_template_vis": dirs["hybrid_templates"] / variant.name / f"{image_id}_hybrid_template_vis.png",
        "semantic_watermarked": dirs["semantic_watermarked"] / variant.name / f"{image_id}_semantic_watermarked.png",
        "semantic_tampered": dirs["semantic_tampered"] / variant.name / f"{image_id}_semantic_tampered.png",
        "recovered_hybrid_template": dirs["recovered_templates"] / variant.name / f"{image_id}_recovered_hybrid_template.png",
        "semantic_pred_mask": dirs["predicted_masks"] / variant.name / f"{image_id}_semantic_pred_mask.png",
        "predicted_regions": dirs["severity_overlays"] / variant.name / f"{image_id}_predicted_severity_regions.json",
        "severity_overlay": dirs["severity_overlays"] / variant.name / f"{image_id}_severity_overlay.png",
        "contact_row": dirs["severity_overlays"] / variant.name / f"{image_id}_contact_row.png",
        "report": dirs["reports"] / f"{variant.name}_{image_id}_report.md",
        "metrics": dirs["reports"] / f"{variant.name}_{image_id}_metrics.json",
    }
    if paths["metrics"].exists() and not args.overwrite:
        row = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        return row, paths["contact_row"] if paths["contact_row"].exists() else None

    original_img = load_rgb(original_path)
    original = np.asarray(original_img, dtype=np.uint8)
    gt_mask = load_mask(gt_mask_path, args.mask_threshold)
    if int(gt_mask.sum()) == 0:
        raise ValueError(f"GT mask has no foreground pixels: {gt_mask_path}")

    prior, prior_quantized, prior_meta = compute_image_prior(original)
    save_prior(prior, prior_quantized, image_id, dirs)
    prior_idx = prior_index_map(prior_quantized)

    base = original_editguard_base_template(512, 512)
    template, payload_prior, normalization = build_hybrid_template(base, prior_quantized, variant)
    save_template(template, paths["hybrid_template"], paths["hybrid_template_vis"])

    message_bits = message_bits_for(image_id, args.seed, sample_index)
    data = image_to_editguard_batch(original, template)
    data["MES"] = message_bits
    model.feed_data(data)
    model.mes = message_bits
    watermarked = model.image_hiding()
    Image.fromarray(watermarked).save(paths["semantic_watermarked"])

    tampered = deterministic_tamper(watermarked, gt_mask, sample_index)
    Image.fromarray(tampered).save(paths["semantic_tampered"])

    recovered_rgb, recmessage = recover_template(model, tampered, template)
    Image.fromarray(np.round(np.clip(recovered_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(paths["recovered_hybrid_template"])

    error_map = np.abs(template - recovered_rgb).max(axis=2)
    pred_mask = error_map > float(args.delta)
    Image.fromarray(pred_mask.astype(np.uint8) * 255, mode="L").save(paths["semantic_pred_mask"])

    gt_labels, gt_sev_index, gt_regions = gt_eval_severity(gt_mask)
    pred_sev_map, predicted_regions = predicted_severity(pred_mask, prior_idx)
    paths["predicted_regions"].write_text(json.dumps(predicted_regions, ensure_ascii=False, indent=2), encoding="utf-8")
    overlay = severity_overlay(Image.fromarray(tampered), pred_sev_map, predicted_regions)
    overlay.save(paths["severity_overlay"])

    pix = pixel_metrics(gt_mask, pred_mask)
    comp = component_metrics(gt_mask, pred_mask, args.small_area_pixels, args.tiny_area_pixels)
    sev = severity_metrics(gt_sev_index, pred_sev_map, gt_labels, gt_regions)
    copyright_payload = evaluate_copyright(message_bits, recmessage)

    row: dict[str, Any] = {
        "variant": variant.name,
        "variant_family": variant.family,
        "epsilon": variant.epsilon,
        "image_id": image_id,
        "status": "ok",
        "error_message": "",
        "base_template_source": "original_editguard_full_support_blue_secret_rgb_[0,0,1]",
        "normalization": normalization,
        "original_path": str(original_path.resolve()),
        "gt_mask_path": str(gt_mask_path.resolve()),
        "prior_path": str(paths["prior"].resolve()),
        "prior_vis_path": str(paths["prior_vis"].resolve()),
        "hybrid_template_path": str(paths["hybrid_template"].resolve()),
        "hybrid_template_vis_path": str(paths["hybrid_template_vis"].resolve()),
        "semantic_watermarked_path": str(paths["semantic_watermarked"].resolve()),
        "semantic_tampered_path": str(paths["semantic_tampered"].resolve()),
        "recovered_hybrid_template_path": str(paths["recovered_hybrid_template"].resolve()),
        "semantic_pred_mask_path": str(paths["semantic_pred_mask"].resolve()),
        "predicted_severity_regions_path": str(paths["predicted_regions"].resolve()),
        "severity_overlay_path": str(paths["severity_overlay"].resolve()),
        "report_path": str(paths["report"].resolve()),
        "copyright_eval_available": copyright_payload["copyright_eval_available"],
        "semantic_template_levels": 5,
        "extra_watermark_channel": False,
        "robust_payload_increase": 0,
        "psnr": psnr(original, watermarked),
        "ssim": ssim_rgb(original, watermarked),
        "bit_accuracy": copyright_payload["bit_accuracy"],
        "ber": copyright_payload["ber"],
        "region_severity_accuracy": sev["region_severity_accuracy"],
        "pixel_severity_accuracy": sev["pixel_severity_accuracy"],
    }
    row.update(pix)
    row.update(comp)

    if args.save_visualizations:
        save_contact_row(
            original_img,
            Image.open(paths["prior_vis"]).convert("RGB"),
            Image.open(paths["hybrid_template_vis"]).convert("RGB"),
            Image.fromarray(watermarked),
            Image.open(paths["recovered_hybrid_template"]).convert("RGB"),
            pred_mask,
            overlay,
            row,
            paths["contact_row"],
        )
    write_report(paths["report"], row, prior_meta)
    paths["metrics"].write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row, paths["contact_row"] if paths["contact_row"].exists() else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def mean(values: list[Any]) -> float:
    vals = []
    for value in values:
        if value in ("", None):
            continue
        try:
            vals.append(float(value))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else 0.0


def summarize_by_variant(rows: list[dict[str, Any]], variants: list[Variant]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in variants:
        group_rows = [row for row in rows if row.get("variant") == variant.name and row.get("status") == "ok"]
        out[variant.name] = {
            "variant_family": variant.family,
            "epsilon": variant.epsilon,
            "evaluated_samples": len(group_rows),
            "mean_metrics": {field: mean([row.get(field) for row in group_rows]) for field in METRIC_FIELDS},
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
    rows = [Image.open(path).convert("RGB").resize((980, 165), BICUBIC) for path in selected]
    if not rows:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}
    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(rows)}


def load_summary_metric(path: Path, key: str | None = None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key is None:
        return payload
    return payload.get(key)


def best_stage7b(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    variants = summary.get("variants", {})
    best_name = summary.get("best_variants", {}).get("best_balanced")
    if best_name and best_name in variants:
        return {"name": best_name, **variants[best_name]}
    if not variants:
        return None
    best_name = max(variants, key=lambda name: variants[name]["mean_metrics"].get("iou", 0.0))
    return {"name": best_name, **variants[best_name]}


def pick_best_7c(summary_by_variant: dict[str, Any]) -> dict[str, str | None]:
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
            0.45 * metric(name, "iou")
            + 0.25 * metric(name, "dice")
            + 0.20 * metric(name, "region_severity_accuracy")
            + 0.10 * metric(name, "pixel_severity_accuracy")
        ),
    )
    return {"best_localization": best_loc, "best_severity": best_sev, "best_balanced": best_bal}


def write_overview(
    output_path: Path,
    stage7_metrics: dict[str, Any] | None,
    stage7b_best: dict[str, Any] | None,
    summary_by_variant: dict[str, Any],
    best_7c: dict[str, str | None],
) -> None:
    lines = [
        "# Stage 7C Non-Oracle Semantic Template Validation",
        "",
        "Stage 7B is an oracle upper bound because its template support used GT-mask-derived information.",
        "Stage 7C removes this oracle dependency.",
        "GT masks are used only for tamper generation and evaluation.",
        "The current severity prior is lightweight and image-derived, not object-semantic.",
        "If Stage 7C retains good localization while producing non-trivial severity accuracy, it becomes the realistic candidate method.",
        "",
        "## References",
        "",
    ]
    if stage7_metrics:
        lines.append(
            f"- Stage 7 pure 5-level: IoU={stage7_metrics.get('iou')}, Dice={stage7_metrics.get('dice')}, "
            f"region_severity_accuracy={stage7_metrics.get('region_severity_accuracy')}, fragmentation_ratio={stage7_metrics.get('fragmentation_ratio')}"
        )
    if stage7b_best:
        m = stage7b_best["mean_metrics"]
        lines.append(
            f"- Stage 7B oracle best ({stage7b_best['name']}): IoU={m.get('iou')}, Dice={m.get('dice')}, "
            f"region_severity_accuracy={m.get('region_severity_accuracy')}, fragmentation_ratio={m.get('fragmentation_ratio')}"
        )
    lines.extend(["", "## Stage 7C Variants", ""])
    lines.append("| Variant | IoU | Dice | Precision | Recall | Region Sev Acc | Pixel Sev Acc | Frag Ratio | PSNR | SSIM | Bit Acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, payload in summary_by_variant.items():
        m = payload["mean_metrics"]
        lines.append(
            f"| {name} | {m['iou']:.6f} | {m['dice']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | "
            f"{m['region_severity_accuracy']:.6f} | {m['pixel_severity_accuracy']:.6f} | "
            f"{m['fragmentation_ratio']:.6f} | {m['psnr']:.6f} | {m['ssim']:.6f} | {m['bit_accuracy']:.6f} |"
        )
    base_metrics = summary_by_variant.get("base_only_control", {}).get("mean_metrics", {})
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Stage 7C base_only_control: IoU={base_metrics.get('iou')}, Dice={base_metrics.get('dice')}, fragmentation_ratio={base_metrics.get('fragmentation_ratio')}",
            f"- Best Stage 7C localization: {best_7c.get('best_localization')}",
            f"- Best Stage 7C severity prior retention: {best_7c.get('best_severity')}",
            f"- Best Stage 7C balanced realistic candidate: {best_7c.get('best_balanced')}",
            "",
            "No Stable Diffusion, SAM, CLIP, DINO, OmniGuard, training, third watermark channel, or robust payload increase is used.",
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

    model = load_editguard_model(project_root, project_root / args.opt, project_root / args.ckpt)
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    vis_paths: list[Path] = []
    total_cases = len(samples) * len(variants)
    done = 0
    for variant in variants:
        for sample_index, sample_dir in enumerate(samples):
            done += 1
            try:
                row, vis_path = process_case(model, sample_dir, sample_index, variant, dirs, args)
                rows.append(row)
                if vis_path is not None:
                    vis_paths.append(vis_path)
            except Exception as exc:
                failure = {
                    "variant": variant.name,
                    "image_id": sample_dir.name,
                    "failure_stage": "stage7c_process_case",
                    "error_message": str(exc),
                }
                failed.append(failure)
                rows.append(
                    {
                        "variant": variant.name,
                        "variant_family": variant.family,
                        "epsilon": variant.epsilon,
                        "image_id": sample_dir.name,
                        "status": "failed",
                        "error_message": str(exc),
                        "semantic_template_levels": 5,
                        "extra_watermark_channel": False,
                        "robust_payload_increase": 0,
                    }
                )
            if done % 10 == 0 or done == total_cases:
                print(f"Stage 7C progress: {done}/{total_cases} cases processed, failed={len(failed)}", flush=True)

    contact_sheet = build_contact_sheet(vis_paths, output_dir / "stage7c_contact_sheet.png", args.contact_sheet_samples)
    write_csv(output_dir / "per_sample_stage7c_metrics.csv", rows)
    (output_dir / "stage7c_failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_by_variant = summarize_by_variant(rows, variants)
    best_7c = pick_best_7c(summary_by_variant)
    stage7_payload = load_summary_metric((project_root / args.stage7_summary).resolve(), "mean_metrics")
    stage7b_payload = load_summary_metric((project_root / args.stage7b_summary).resolve())
    stage7b_best = best_stage7b(stage7b_payload)
    summary = {
        "stage": "stage7c_non_oracle_semantic_template",
        "total_cases": len(rows),
        "evaluated_cases": len([row for row in rows if row.get("status") == "ok"]),
        "failed_cases": len(failed),
        "max_samples": args.max_samples,
        "delta": args.delta,
        "base_template_source": "original_editguard_full_support_blue_secret_rgb_[0,0,1]",
        "gt_mask_usage": ["deterministic_tamper_generation", "evaluation_only"],
        "gt_mask_used_for_template_generation": False,
        "gt_mask_used_for_severity_prior_generation": False,
        "gt_mask_used_for_watermark_payload_construction": False,
        "variants": summary_by_variant,
        "stage7_pure_5_level_reference": stage7_payload,
        "stage7b_oracle_best_reference": stage7b_best,
        "stage7c_base_only_control": summary_by_variant.get("base_only_control"),
        "best_variants": best_7c,
        "payload": {
            "semantic_template_levels": 5,
            "extra_watermark_channel": False,
            "robust_payload_increase": 0,
        },
        "contact_sheet": contact_sheet,
        "constraints": {
            "no_training": True,
            "no_stable_diffusion": True,
            "no_sam_clip_dino": True,
            "no_omniguard_switch": True,
            "gt_mask_used_as_predicted_mask": False,
            "original_editguard_checkpoint_overwritten": False,
            "previous_stage_outputs_modified": False,
        },
    }
    (output_dir / "stage7c_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_overview(output_dir / "stage7c_overview.md", stage7_payload, stage7b_best, summary_by_variant, best_7c)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Saved metrics: {output_dir / 'per_sample_stage7c_metrics.csv'}")
    print(f"Saved summary: {output_dir / 'stage7c_summary.json'}")
    print(f"Saved overview: {output_dir / 'stage7c_overview.md'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
