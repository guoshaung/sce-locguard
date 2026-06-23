from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

SEVERITY_CODES = [0.0, 0.25, 0.5, 0.75, 1.0]
FOREGROUND_CODES = [0.25, 0.5, 0.75, 1.0]
CODE_TO_LABEL = {
    0.0: "background",
    0.25: "low",
    0.5: "medium",
    0.75: "high",
    1.0: "critical",
}
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
    "image_id",
    "status",
    "error_message",
    "original_path",
    "gt_mask_path",
    "semantic_template_path",
    "semantic_template_vis_path",
    "severity_regions_path",
    "semantic_watermarked_path",
    "semantic_tampered_path",
    "recovered_semantic_template_path",
    "semantic_pred_mask_path",
    "predicted_severity_regions_path",
    "severity_overlay_path",
    "report_path",
    "copyright_eval_available",
    "severity_confusion_matrix",
    "semantic_template_levels",
    "extra_watermark_channel",
    "robust_payload_increase",
] + METRIC_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 7 semantic-structured fragile localization template prototype."
    )
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--input_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage7_semantic_structured_template")
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
        "semantic_templates": output_dir / "semantic_templates",
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


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((512, 512), BICUBIC)


def load_mask(path: Path, threshold: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((512, 512), NEAREST)
    return np.asarray(mask, dtype=np.uint8) > threshold


def image_to_editguard_batch(image: np.ndarray, secret_template: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    img_gt = image.astype(np.float32) / 255.0
    img_gt = img_gt[:, :, [2, 1, 0]]
    img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float().unsqueeze(0)
    img_gt = torch.nn.functional.interpolate(img_gt, size=(512, 512), mode="nearest", align_corners=None)
    img_gt = img_gt.unsqueeze(0)

    _, t, _, h, w = img_gt.shape
    if secret_template is None:
        secret = np.zeros((t, h, w, 3), dtype=np.float32)
        secret[..., 2] = 1.0
    else:
        if secret_template.ndim == 2:
            secret_rgb = np.repeat(secret_template[:, :, None], 3, axis=2)
        else:
            secret_rgb = secret_template
        secret = np.expand_dims(secret_rgb.astype(np.float32), axis=0)
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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def center_score(cx: float, cy: float, width: int, height: int) -> float:
    image_cx = (width - 1) / 2.0
    image_cy = (height - 1) / 2.0
    max_dist = math.sqrt(image_cx * image_cx + image_cy * image_cy)
    dist = math.sqrt((cx - image_cx) ** 2 + (cy - image_cy) ** 2)
    return float(np.clip(1.0 - safe_div(dist, max_dist), 0.0, 1.0))


def severity_from_score(score: float) -> tuple[float, str]:
    if score < 0.25:
        return 0.25, "low"
    if score < 0.50:
        return 0.50, "medium"
    if score < 0.75:
        return 0.75, "high"
    return 1.0, "critical"


def nearest_code(value: float, include_background: bool = False) -> float:
    codes = SEVERITY_CODES if include_background else FOREGROUND_CODES
    return min(codes, key=lambda code: abs(float(value) - code))


def generate_semantic_template(gt_mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    height, width = gt_mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_mask.astype(np.uint8), connectivity=8)
    template = np.zeros((height, width), dtype=np.float32)
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
        severity_score = float(0.7 * area_score + 0.3 * c_score)
        code, label = severity_from_score(severity_score)
        template[labels == idx] = code
        regions.append(
            {
                "region_id": int(idx),
                "bbox": [x, y, x + w, y + h],
                "area_pixels": area,
                "area_ratio": area_ratio,
                "centroid": [float(cx), float(cy)],
                "area_score": area_score,
                "center_score": c_score,
                "severity_score": severity_score,
                "severity_code": code,
                "severity_label": label,
                "pseudo_label_rule": "0.7*area_score + 0.3*center_score",
            }
        )
    return template, regions, labels


def save_template_images(template: np.ndarray, template_path: Path, vis_path: Path) -> None:
    Image.fromarray(np.round(template * 255.0).astype(np.uint8), mode="L").save(template_path)
    vis = np.zeros((*template.shape, 3), dtype=np.uint8)
    for code, label in CODE_TO_LABEL.items():
        vis[np.isclose(template, code)] = LABEL_TO_COLOR[label]
    Image.fromarray(vis, mode="RGB").save(vis_path)


def recover_semantic_template(model: Any, tampered: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    from models.IBSN import iwt

    data = image_to_editguard_batch(tampered, template)
    data["MES"] = None
    model.feed_data(data)
    model.netG.eval()
    with torch.no_grad():
        b, t, c, h, w = model.real_H.shape
        center = t // 2
        intval = model.gop // 2
        b, n, t, c, h, w = model.ref_L.shape
        host = model.real_H[:, center - intval : center + intval + 1]
        y = model.Quantization(host.squeeze(1))
        _out_x, out_x_h, _out_z, recmessage = model.netG(x=y, rev=True)
        out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
        out_x_h = torch.stack(out_x_h, dim=1)
        out_x_h = out_x_h.reshape(-1, 1, model.gop, 3, h, w)
        rec_loc = out_x_h[:, :, model.gop // 2]
        rec_rgb = rec_loc[0, 0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        rec_gray = rec_rgb.mean(axis=2).astype(np.float32)
        return rec_gray, rec_rgb, recmessage


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
        return {
            "copyright_eval_available": False,
            "bit_accuracy": "",
            "ber": "",
            "copyright_error_message": str(exc),
        }


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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
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


def build_predicted_severity(template: np.ndarray, pred_mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), connectivity=8)
    pred_severity = np.zeros_like(template, dtype=np.float32)
    regions: list[dict[str, Any]] = []
    image_area = float(template.size)
    for idx in range(1, num_labels):
        region = labels == idx
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        q_value = float(template[region].mean()) if area else 0.0
        code = nearest_code(q_value, include_background=False)
        label = CODE_TO_LABEL[code]
        pred_severity[region] = code
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
                "q_mean_template_inside_region": q_value,
                "predicted_severity_code": code,
                "predicted_severity_label": label,
            }
        )
    return pred_severity, regions


def severity_metrics(
    template: np.ndarray,
    pred_severity: np.ndarray,
    gt_labels: np.ndarray,
    severity_regions: list[dict[str, Any]],
) -> dict[str, Any]:
    code_to_index = {code: idx for idx, code in enumerate(SEVERITY_CODES)}
    true_indices = np.vectorize(lambda v: code_to_index[nearest_code(float(v), include_background=True)])(template)
    pred_indices = np.vectorize(lambda v: code_to_index[nearest_code(float(v), include_background=True)])(pred_severity)
    confusion = np.zeros((len(SEVERITY_CODES), len(SEVERITY_CODES)), dtype=np.int64)
    for true_idx, pred_idx in zip(true_indices.reshape(-1), pred_indices.reshape(-1)):
        confusion[int(true_idx), int(pred_idx)] += 1

    foreground = template > 0
    pixel_accuracy = safe_div(int((true_indices[foreground] == pred_indices[foreground]).sum()), int(foreground.sum()))

    correct_regions = 0
    total_regions = 0
    for region in severity_regions:
        region_mask = gt_labels == int(region["region_id"])
        if not np.any(region_mask):
            continue
        total_regions += 1
        true_code = float(region["severity_code"])
        pred_codes = pred_severity[region_mask]
        if pred_codes.size == 0:
            pred_code = 0.0
        else:
            values, counts = np.unique(np.round(pred_codes * 4.0) / 4.0, return_counts=True)
            pred_code = float(values[int(np.argmax(counts))])
        if abs(pred_code - true_code) < 1e-6:
            correct_regions += 1
    region_accuracy = safe_div(correct_regions, total_regions)
    return {
        "region_severity_accuracy": region_accuracy,
        "pixel_severity_accuracy": pixel_accuracy,
        "severity_confusion_matrix": confusion.tolist(),
        "severity_confusion_labels": [CODE_TO_LABEL[code] for code in SEVERITY_CODES],
    }


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


def severity_overlay(tampered: Image.Image, pred_severity: np.ndarray, regions: list[dict[str, Any]]) -> Image.Image:
    base = np.asarray(tampered.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(base)
    active = pred_severity > 0
    for code, label in CODE_TO_LABEL.items():
        if code == 0.0:
            continue
        color[np.isclose(pred_severity, code)] = LABEL_TO_COLOR[label]
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


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (180, 180)) -> Image.Image:
    header_h = 30
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 9), title[:26], fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def save_contact_visual(
    original: Image.Image,
    template_vis: Image.Image,
    watermarked: Image.Image,
    tampered: Image.Image,
    recovered_template: Image.Image,
    pred_mask: np.ndarray,
    overlay: Image.Image,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    panels = [
        title_panel(original, f"{row['image_id']} original"),
        title_panel(template_vis, "S_sem"),
        title_panel(watermarked, f"PSNR {float(row['psnr']):.2f}"),
        title_panel(tampered, "tampered"),
        title_panel(recovered_template, "S_hat"),
        title_panel(mask_image(pred_mask), f"IoU {float(row['iou']):.3f}"),
        title_panel(overlay, f"sev {float(row['region_severity_accuracy']):.2f}"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_sample_report(
    report_path: Path,
    row: dict[str, Any],
    severity_regions: list[dict[str, Any]],
    predicted_regions: list[dict[str, Any]],
    severity_metrics_payload: dict[str, Any],
) -> None:
    lines = [
        f"# Stage 7 Report: {row['image_id']}",
        "",
        "## Paths",
        "",
        f"- original image path: {row['original_path']}",
        f"- gt mask path: {row['gt_mask_path']}",
        f"- semantic template path: {row['semantic_template_path']}",
        f"- recovered semantic template path: {row['recovered_semantic_template_path']}",
        f"- predicted mask path: {row['semantic_pred_mask_path']}",
        "",
        "## Severity Labels",
        "",
        f"- pseudo severity regions: {len(severity_regions)}",
        f"- predicted severity regions: {len(predicted_regions)}",
        "- pseudo labels are generated by connected-component area and center-proximity rules.",
        "",
        "## Localization Metrics",
        "",
        f"- IoU: {row['iou']}",
        f"- Dice: {row['dice']}",
        f"- Precision: {row['precision']}",
        f"- Recall: {row['recall']}",
        f"- MAE: {row['mae']}",
        "",
        "## Severity Metrics",
        "",
        f"- region_severity_accuracy: {row['region_severity_accuracy']}",
        f"- pixel_severity_accuracy: {row['pixel_severity_accuracy']}",
        f"- severity_confusion_matrix: {json.dumps(severity_metrics_payload['severity_confusion_matrix'])}",
        "",
        "## Copyright And Quality",
        "",
        f"- bit accuracy: {row['bit_accuracy']}",
        f"- BER: {row['ber']}",
        f"- PSNR: {row['psnr']}",
        f"- SSIM: {row['ssim']}",
        "",
        "## Failure Reason",
        "",
        row.get("error_message") or "none",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_sample(
    model: Any,
    sample_dir: Path,
    sample_index: int,
    dirs: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path | None]:
    image_id = sample_dir.name
    paths = {
        "original": sample_dir / "original.png",
        "gt_mask": sample_dir / "gt_mask.png",
        "semantic_template": dirs["semantic_templates"] / f"{image_id}_semantic_template.png",
        "semantic_template_vis": dirs["semantic_templates"] / f"{image_id}_semantic_template_vis.png",
        "severity_regions": dirs["semantic_templates"] / f"{image_id}_severity_regions.json",
        "semantic_watermarked": dirs["semantic_watermarked"] / f"{image_id}_semantic_watermarked.png",
        "semantic_tampered": dirs["semantic_tampered"] / f"{image_id}_semantic_tampered.png",
        "recovered_template": dirs["recovered_templates"] / f"{image_id}_recovered_semantic_template.png",
        "semantic_pred_mask": dirs["predicted_masks"] / f"{image_id}_semantic_pred_mask.png",
        "predicted_severity_regions": dirs["severity_overlays"] / f"{image_id}_predicted_severity_regions.json",
        "severity_overlay": dirs["severity_overlays"] / f"{image_id}_severity_overlay.png",
        "contact_visual": dirs["severity_overlays"] / f"{image_id}_stage7_contact_row.png",
        "report": dirs["reports"] / f"{image_id}_stage7_report.md",
        "metrics": dirs["reports"] / f"{image_id}_stage7_metrics.json",
    }
    if paths["metrics"].exists() and not args.overwrite:
        return json.loads(paths["metrics"].read_text(encoding="utf-8")), paths["contact_visual"] if paths["contact_visual"].exists() else None

    row: dict[str, Any] = {
        "image_id": image_id,
        "status": "ok",
        "error_message": "",
        "original_path": str(paths["original"].resolve()),
        "gt_mask_path": str(paths["gt_mask"].resolve()),
        "semantic_template_path": str(paths["semantic_template"].resolve()),
        "semantic_template_vis_path": str(paths["semantic_template_vis"].resolve()),
        "severity_regions_path": str(paths["severity_regions"].resolve()),
        "semantic_watermarked_path": str(paths["semantic_watermarked"].resolve()),
        "semantic_tampered_path": str(paths["semantic_tampered"].resolve()),
        "recovered_semantic_template_path": str(paths["recovered_template"].resolve()),
        "semantic_pred_mask_path": str(paths["semantic_pred_mask"].resolve()),
        "predicted_severity_regions_path": str(paths["predicted_severity_regions"].resolve()),
        "severity_overlay_path": str(paths["severity_overlay"].resolve()),
        "report_path": str(paths["report"].resolve()),
        "semantic_template_levels": 5,
        "extra_watermark_channel": False,
        "robust_payload_increase": 0,
    }

    for required in [paths["original"], paths["gt_mask"]]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required Stage 1 input: {required}")

    original_img = load_rgb(paths["original"])
    original = np.asarray(original_img, dtype=np.uint8)
    gt_mask = load_mask(paths["gt_mask"], args.mask_threshold)
    if int(gt_mask.sum()) == 0:
        raise ValueError(f"GT mask has no foreground pixels: {paths['gt_mask']}")

    template, severity_regions, gt_labels = generate_semantic_template(gt_mask)
    save_template_images(template, paths["semantic_template"], paths["semantic_template_vis"])
    paths["severity_regions"].write_text(json.dumps(severity_regions, ensure_ascii=False, indent=2), encoding="utf-8")

    message_bits = message_bits_for(image_id, args.seed, sample_index)
    data = image_to_editguard_batch(original, template)
    data["MES"] = message_bits
    model.feed_data(data)
    model.mes = message_bits
    semantic_watermarked = model.image_hiding()
    Image.fromarray(semantic_watermarked).save(paths["semantic_watermarked"])

    semantic_tampered = deterministic_tamper(semantic_watermarked, gt_mask, sample_index)
    Image.fromarray(semantic_tampered).save(paths["semantic_tampered"])

    recovered_gray, _recovered_rgb, recmessage = recover_semantic_template(model, semantic_tampered, template)
    Image.fromarray(np.round(np.clip(recovered_gray, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(paths["recovered_template"])

    error_map = np.abs(template - recovered_gray)
    pred_mask = error_map > float(args.delta)
    Image.fromarray(pred_mask.astype(np.uint8) * 255, mode="L").save(paths["semantic_pred_mask"])

    pred_severity, predicted_regions = build_predicted_severity(template, pred_mask)
    paths["predicted_severity_regions"].write_text(
        json.dumps(predicted_regions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    overlay = severity_overlay(Image.fromarray(semantic_tampered), pred_severity, predicted_regions)
    overlay.save(paths["severity_overlay"])

    pix = pixel_metrics(gt_mask, pred_mask)
    comp = component_metrics(gt_mask, pred_mask, args.small_area_pixels, args.tiny_area_pixels)
    sev = severity_metrics(template, pred_severity, gt_labels, severity_regions)
    copyright_payload = evaluate_copyright(message_bits, recmessage)

    row.update(pix)
    row.update(comp)
    row.update(
        {
            "psnr": psnr(original, semantic_watermarked),
            "ssim": ssim_rgb(original, semantic_watermarked),
            "region_severity_accuracy": sev["region_severity_accuracy"],
            "pixel_severity_accuracy": sev["pixel_severity_accuracy"],
            "severity_confusion_matrix": json.dumps(sev["severity_confusion_matrix"]),
            "copyright_eval_available": copyright_payload["copyright_eval_available"],
            "bit_accuracy": copyright_payload["bit_accuracy"],
            "ber": copyright_payload["ber"],
        }
    )

    if args.save_visualizations:
        save_contact_visual(
            original_img,
            Image.open(paths["semantic_template_vis"]).convert("RGB"),
            Image.fromarray(semantic_watermarked),
            Image.fromarray(semantic_tampered),
            Image.open(paths["recovered_template"]).convert("L").convert("RGB"),
            pred_mask,
            overlay,
            row,
            paths["contact_visual"],
        )
    write_sample_report(paths["report"], row, severity_regions, predicted_regions, sev)
    paths["metrics"].write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row, paths["contact_visual"] if paths["contact_visual"].exists() else None


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


def build_contact_sheet(vis_paths: list[Path], output_path: Path, max_items: int) -> dict[str, Any]:
    existing = [path for path in vis_paths if path and path.exists()]
    if len(existing) > max_items:
        indices = sorted({round(i * (len(existing) - 1) / (max_items - 1)) for i in range(max_items)})
        selected = [existing[index] for index in indices]
        sampling = "even"
    else:
        selected = existing
        sampling = "all"
    rows = [Image.open(path).convert("RGB").resize((980, 168), BICUBIC) for path in selected]
    if not rows:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}
    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(rows)}


def aggregate_confusion(rows: list[dict[str, Any]]) -> list[list[int]]:
    total = np.zeros((len(SEVERITY_CODES), len(SEVERITY_CODES)), dtype=np.int64)
    for row in rows:
        payload = row.get("severity_confusion_matrix")
        if not payload:
            continue
        total += np.asarray(json.loads(payload), dtype=np.int64)
    return total.tolist()


def write_overview(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 7 Overview",
        "",
        "This is a lightweight prototype.",
        "It does not add a third watermark channel.",
        "It converts the original binary fragile localization template into a semantic-structured template.",
        "It enables tamper localization plus severity indication.",
        "The current severity labels are pseudo labels generated by connected-component rules, not real semantic labels from SAM/CLIP/DINO.",
        "Future work can replace pseudo severity with real semantic object-level labels.",
        "",
        "## Summary",
        "",
        f"- total_samples: {summary['total_samples']}",
        f"- evaluated_samples: {summary['evaluated_samples']}",
        f"- failed_samples: {summary['failed_samples']}",
        f"- semantic_template_levels: {summary['payload']['semantic_template_levels']}",
        f"- extra_watermark_channel: {summary['payload']['extra_watermark_channel']}",
        f"- robust_payload_increase: {summary['payload']['robust_payload_increase']}",
        "",
        "## Mean Metrics",
        "",
    ]
    for key, value in summary["mean_metrics"].items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: list[dict[str, Any]], failed: list[dict[str, Any]], contact_sheet: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "stage": "stage7_semantic_structured_template",
        "total_samples": len(rows) + len(failed),
        "evaluated_samples": len(ok_rows),
        "failed_samples": len(failed),
        "delta": args.delta,
        "max_samples": args.max_samples,
        "mean_metrics": {field: mean([row.get(field) for row in ok_rows]) for field in METRIC_FIELDS},
        "severity_confusion_matrix_sum": aggregate_confusion(ok_rows),
        "severity_confusion_labels": [CODE_TO_LABEL[code] for code in SEVERITY_CODES],
        "payload": {
            "semantic_template_levels": 5,
            "extra_watermark_channel": False,
            "robust_payload_increase": 0,
        },
        "contact_sheet": contact_sheet,
        "prototype_conclusions": [
            "This is a lightweight prototype.",
            "It does not add a third watermark channel.",
            "It converts the original binary fragile localization template into a semantic-structured template.",
            "It enables tamper localization plus severity indication.",
            "The current severity labels are pseudo labels generated by connected-component rules, not real semantic labels from SAM/CLIP/DINO.",
            "Future work can replace pseudo severity with real semantic object-level labels.",
        ],
        "constraints": {
            "no_stable_diffusion": True,
            "no_sam_clip_dino": True,
            "no_omniguard_switch": True,
            "gt_mask_used_as_predicted_mask": False,
            "original_editguard_checkpoint_overwritten": False,
            "previous_stage_outputs_modified": False,
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    dirs = ensure_dirs(output_dir)

    samples = sample_dirs(input_dir, args.max_samples)
    if not samples:
        raise FileNotFoundError(f"No Stage 1 generated samples found under {input_dir}")

    model = load_editguard_model(project_root, project_root / args.opt, project_root / args.ckpt)
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    vis_paths: list[Path] = []

    for idx, sample_dir in enumerate(samples, start=1):
        try:
            row, vis_path = process_sample(model, sample_dir, idx - 1, dirs, args)
            rows.append(row)
            if vis_path is not None:
                vis_paths.append(vis_path)
        except Exception as exc:
            image_id = sample_dir.name
            failure = {
                "image_id": image_id,
                "failure_stage": "stage7_process_sample",
                "error_message": str(exc),
            }
            failed.append(failure)
            report_path = dirs["reports"] / f"{image_id}_stage7_report.md"
            report_path.write_text(
                f"# Stage 7 Report: {image_id}\n\n## Failure Reason\n\n{exc}\n",
                encoding="utf-8",
            )
            rows.append(
                {
                    "image_id": image_id,
                    "status": "failed",
                    "error_message": str(exc),
                    "report_path": str(report_path.resolve()),
                    "semantic_template_levels": 5,
                    "extra_watermark_channel": False,
                    "robust_payload_increase": 0,
                }
            )
        if idx % 5 == 0 or idx == len(samples):
            print(f"Stage 7 progress: {idx}/{len(samples)} samples processed, failed={len(failed)}", flush=True)

    contact_sheet = build_contact_sheet(vis_paths, output_dir / "stage7_contact_sheet.png", args.contact_sheet_samples)
    write_csv(output_dir / "per_sample_stage7_metrics.csv", rows)
    (output_dir / "stage7_failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(rows, failed, contact_sheet, args)
    (output_dir / "stage7_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_overview(dirs["reports"] / "stage7_overview.md", summary)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Saved metrics: {output_dir / 'per_sample_stage7_metrics.csv'}")
    print(f"Saved summary: {output_dir / 'stage7_summary.json'}")
    print(f"Saved overview: {dirs['reports'] / 'stage7_overview.md'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
