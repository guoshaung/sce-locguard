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

METRIC_NAMES = [
    "psnr",
    "ssim",
    "iou",
    "dice",
    "precision",
    "recall",
    "mae",
    "pred_mask_area_ratio",
    "gt_mask_area_ratio",
    "num_pred_regions",
    "num_gt_regions",
    "fragmentation_ratio",
    "small_pred_region_ratio",
    "tiny_pred_region_ratio",
    "semantic_compactness",
    "bit_accuracy",
    "ber",
]

CSV_FIELDS = [
    "image_id",
    "status",
    "error_message",
    "original_path",
    "gt_mask_path",
    "baseline_pred_mask_path",
    "candidate_pred_mask_path",
    "candidate_watermarked_path",
    "candidate_tampered_path",
]
for metric_name in METRIC_NAMES:
    CSV_FIELDS.append(f"baseline_{metric_name}")
    CSV_FIELDS.append(f"candidate_{metric_name}")
    if metric_name not in ("num_gt_regions", "gt_mask_area_ratio"):
        CSV_FIELDS.append(f"delta_{metric_name}")
CSV_FIELDS.extend(
    [
        "baseline_copyright_eval_available",
        "candidate_copyright_eval_available",
        "improved_iou",
        "improved_fragmentation",
        "improved_both",
        "worsened_iou",
        "worsened_fragmentation",
        "stable_bit_accuracy",
    ]
)

DELTA_FIELDS = [
    "image_id",
    "delta_iou",
    "delta_dice",
    "delta_psnr",
    "delta_fragmentation_ratio",
    "delta_small_pred_region_ratio",
    "delta_tiny_pred_region_ratio",
    "delta_bit_accuracy",
    "improved_iou",
    "improved_fragmentation",
    "improved_both",
    "worsened_iou",
    "worsened_fragmentation",
    "stable_bit_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale Stage 4C best residual-floor suppression candidate to full valAGE-Set."
    )
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--input_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage4c_scale_best_candidate")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--recovery_threshold", type=float, default=0.2)
    parser.add_argument("--small_area_pixels", type=int, default=100)
    parser.add_argument("--tiny_area_pixels", type=int, default=25)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--overwrite", action="store_true")
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


def load_sample_ids(project_root: Path, input_dir: Path, split: str, start: int, max_samples: int) -> list[str]:
    split_file = project_root / "dataset" / ("sep_vallist.txt" if split == "val" else "sep_testlist.txt")
    if split_file.exists():
        ids = [Path(line.strip()).stem for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        ids = sorted(path.name for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit())
    ids = [image_id for image_id in ids if (input_dir / image_id).is_dir()]
    return ids[start : start + max_samples]


def image_to_editguard_batch(image: np.ndarray) -> dict[str, torch.Tensor]:
    img_gt = image.astype(np.float32) / 255.0
    img_gt = img_gt[:, :, [2, 1, 0]]
    img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float().unsqueeze(0)
    img_gt = torch.nn.functional.interpolate(img_gt, size=(512, 512), mode="nearest", align_corners=None)
    img_gt = img_gt.unsqueeze(0)

    _, t, _, h, w = img_gt.shape
    blue_secret = np.zeros((t, h, w, 3), dtype=np.float32)
    blue_secret[..., 2] = 1.0
    imgs_lq = torch.from_numpy(np.ascontiguousarray(blue_secret)).float().permute(0, 3, 1, 2).unsqueeze(0)
    list_h = torch.stack([imgs_lq], dim=0)
    return {"LQ": list_h, "GT": img_gt}


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
        "pred_mask_area_ratio": safe_div(tp + fp, total),
        "gt_mask_area_ratio": safe_div(tp + fn, total),
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
    return float(np.mean([ssim_channel(original[:, :, idx], watermarked[:, :, idx]) for idx in range(3)]))


def component_metrics(gt: np.ndarray, pred: np.ndarray, small_area_pixels: int, tiny_area_pixels: int) -> dict[str, Any]:
    num_pred, _, pred_stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    num_gt, _, _, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
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


def evaluate_mask_quality(
    original: np.ndarray,
    watermarked: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    small_area_pixels: int,
    tiny_area_pixels: int,
) -> dict[str, Any]:
    metrics = {}
    metrics["psnr"] = psnr(original, watermarked)
    metrics["ssim"] = ssim_rgb(original, watermarked)
    metrics.update(pixel_metrics(gt, pred))
    metrics.update(component_metrics(gt, pred, small_area_pixels, tiny_area_pixels))
    return metrics


def evaluate_copyright(message_bits: np.ndarray, recmessage: Any) -> dict[str, Any]:
    try:
        rec = recmessage.detach().float().cpu().numpy().reshape(-1)
        truth = (message_bits.reshape(-1) > 0).astype(np.uint8)
        pred = (rec > 0.5).astype(np.uint8)
        if pred.shape[0] != truth.shape[0]:
            raise ValueError(f"Recovered bit length mismatch: expected {truth.shape[0]}, got {pred.shape[0]}")
        bit_accuracy = float(np.mean(pred == truth))
        return {
            "copyright_eval_available": True,
            "bit_accuracy": bit_accuracy,
            "ber": float(1.0 - bit_accuracy),
        }
    except Exception as exc:
        return {
            "copyright_eval_available": False,
            "bit_accuracy": "",
            "ber": "",
            "copyright_error_message": str(exc),
        }


def load_mask(path: Path, threshold: int, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, NEAREST)
    return np.asarray(image, dtype=np.uint8) > threshold


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


def error_map_image(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[np.logical_and(pred, gt)] = [255, 255, 255]
    out[np.logical_and(pred, ~gt)] = [255, 80, 80]
    out[np.logical_and(~pred, gt)] = [80, 160, 255]
    return Image.fromarray(out, mode="RGB")


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (180, 180)) -> Image.Image:
    header_h = 30
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 9), title, fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def save_visualization(
    original: Image.Image,
    gt: np.ndarray,
    baseline_pred: np.ndarray,
    candidate_watermarked: Image.Image,
    candidate_tampered: Image.Image,
    candidate_pred: np.ndarray,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    panels = [
        title_panel(original, "original"),
        title_panel(mask_image(gt), "gt_mask"),
        title_panel(mask_image(baseline_pred), f"base IoU {float(row['baseline_iou']):.3f}"),
        title_panel(mask_image(candidate_pred), f"cand IoU {float(row['candidate_iou']):.3f}"),
        title_panel(candidate_watermarked, f"cand PSNR {float(row['candidate_psnr']):.2f}"),
        title_panel(candidate_tampered, "candidate tampered"),
        title_panel(error_map_image(gt, candidate_pred), f"cand frag {float(row['candidate_fragmentation_ratio']):.1f}"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def outputs_complete(sample_out: Path) -> bool:
    required = [
        "baseline_pred_mask.png",
        "candidate_pred_mask.png",
        "candidate_watermarked.png",
        "candidate_tampered.png",
        "candidate_vis.png",
        "metrics.json",
        "meta.json",
    ]
    return all((sample_out / name).exists() for name in required)


def flatten_row(
    image_id: str,
    sample_dir: Path,
    sample_out: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image_id": image_id,
        "status": "ok",
        "error_message": "",
        "original_path": str((sample_dir / "original.png").resolve()),
        "gt_mask_path": str((sample_dir / "gt_mask.png").resolve()),
        "baseline_pred_mask_path": str((sample_out / "baseline_pred_mask.png").resolve()),
        "candidate_pred_mask_path": str((sample_out / "candidate_pred_mask.png").resolve()),
        "candidate_watermarked_path": str((sample_out / "candidate_watermarked.png").resolve()),
        "candidate_tampered_path": str((sample_out / "candidate_tampered.png").resolve()),
        "baseline_copyright_eval_available": baseline.get("copyright_eval_available", False),
        "candidate_copyright_eval_available": candidate.get("copyright_eval_available", False),
    }
    for metric_name in METRIC_NAMES:
        row[f"baseline_{metric_name}"] = baseline.get(metric_name, "")
        row[f"candidate_{metric_name}"] = candidate.get(metric_name, "")
        if metric_name not in ("num_gt_regions", "gt_mask_area_ratio"):
            b_val = baseline.get(metric_name, "")
            c_val = candidate.get(metric_name, "")
            row[f"delta_{metric_name}"] = float(c_val) - float(b_val) if b_val != "" and c_val != "" else ""
    row["improved_iou"] = float(row["delta_iou"]) > 0.0
    row["improved_fragmentation"] = float(row["delta_fragmentation_ratio"]) < 0.0
    row["improved_both"] = bool(row["improved_iou"]) and bool(row["improved_fragmentation"])
    row["worsened_iou"] = float(row["delta_iou"]) < 0.0
    row["worsened_fragmentation"] = float(row["delta_fragmentation_ratio"]) > 0.0
    row["stable_bit_accuracy"] = (
        row["baseline_bit_accuracy"] != ""
        and row["candidate_bit_accuracy"] != ""
        and float(row["candidate_bit_accuracy"]) >= 0.99
        and float(row["candidate_bit_accuracy"]) >= float(row["baseline_bit_accuracy"]) - 1e-9
    )
    return row


def process_sample(
    model: Any,
    image_id: str,
    sample_index: int,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path | None]:
    sample_dir = input_dir / image_id
    sample_out = output_dir / image_id
    visual_path = sample_out / "candidate_vis.png"
    if outputs_complete(sample_out) and not args.overwrite:
        payload = json.loads((sample_out / "metrics.json").read_text(encoding="utf-8"))
        return payload["flat_row"], visual_path

    try:
        original_path = sample_dir / "original.png"
        gt_mask_path = sample_dir / "gt_mask.png"
        baseline_watermarked_path = sample_dir / "watermarked.png"
        baseline_tampered_path = sample_dir / "tampered.png"
        for required in [original_path, gt_mask_path, baseline_watermarked_path, baseline_tampered_path]:
            if not required.exists():
                raise FileNotFoundError(f"Missing required input: {required}")

        sample_out.mkdir(parents=True, exist_ok=True)
        original_img = Image.open(original_path).convert("RGB").resize((512, 512), BICUBIC)
        gt_img = Image.open(gt_mask_path).convert("L").resize((512, 512), NEAREST)
        baseline_watermarked_img = Image.open(baseline_watermarked_path).convert("RGB").resize((512, 512), BICUBIC)
        baseline_tampered_img = Image.open(baseline_tampered_path).convert("RGB").resize((512, 512), BICUBIC)
        original_arr = np.asarray(original_img, dtype=np.uint8)
        gt = np.asarray(gt_img, dtype=np.uint8) > args.mask_threshold
        baseline_watermarked = np.asarray(baseline_watermarked_img, dtype=np.uint8)
        baseline_tampered = np.asarray(baseline_tampered_img, dtype=np.uint8)
        if int(gt.sum()) == 0:
            raise ValueError(f"GT mask is empty: {gt_mask_path}")

        rng_seed = args.seed + int(image_id) if image_id.isdigit() else args.seed + sample_index
        rng = np.random.default_rng(rng_seed)
        message_bits = rng.choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)

        baseline_data = image_to_editguard_batch(baseline_tampered)
        baseline_data["MES"] = None
        model.feed_data(baseline_data)
        baseline_pred_mask, baseline_recmessage = model.image_recovery(args.recovery_threshold)
        baseline_pred = np.asarray(baseline_pred_mask) > 0
        Image.fromarray(baseline_pred.astype(np.uint8) * 255, mode="L").save(sample_out / "baseline_pred_mask.png")
        baseline_metrics = evaluate_mask_quality(
            original_arr,
            baseline_watermarked,
            gt,
            baseline_pred,
            args.small_area_pixels,
            args.tiny_area_pixels,
        )
        baseline_metrics.update(evaluate_copyright(message_bits, baseline_recmessage))

        asem_tensor = torch.from_numpy(gt.astype(np.float32)).float().unsqueeze(0).unsqueeze(0).to(model.device)
        candidate_data = image_to_editguard_batch(original_arr)
        candidate_data["MES"] = message_bits
        model.feed_data(candidate_data)
        model.mes = message_bits
        candidate_watermarked = model.image_hiding(
            asem_map=asem_tensor,
            asem_alpha=args.gamma,
            asem_mode="gt_mask_boost",
            asem_variant="residual_floor_suppression",
            asem_tau=args.tau,
            asem_smooth_sigma=0.0,
        )
        Image.fromarray(candidate_watermarked).save(sample_out / "candidate_watermarked.png")

        candidate_tampered = deterministic_tamper(candidate_watermarked, gt, sample_index)
        Image.fromarray(candidate_tampered).save(sample_out / "candidate_tampered.png")
        candidate_recovery_data = image_to_editguard_batch(candidate_tampered)
        candidate_recovery_data["MES"] = None
        model.feed_data(candidate_recovery_data)
        candidate_pred_mask, candidate_recmessage = model.image_recovery(args.recovery_threshold)
        candidate_pred = np.asarray(candidate_pred_mask) > 0
        Image.fromarray(candidate_pred.astype(np.uint8) * 255, mode="L").save(sample_out / "candidate_pred_mask.png")
        candidate_metrics = evaluate_mask_quality(
            original_arr,
            candidate_watermarked,
            gt,
            candidate_pred,
            args.small_area_pixels,
            args.tiny_area_pixels,
        )
        candidate_metrics.update(evaluate_copyright(message_bits, candidate_recmessage))

        row = flatten_row(image_id, sample_dir, sample_out, baseline_metrics, candidate_metrics)
        save_visualization(
            original_img,
            gt,
            baseline_pred,
            Image.fromarray(candidate_watermarked),
            Image.fromarray(candidate_tampered),
            candidate_pred,
            row,
            visual_path,
        )
        meta = {
            "image_id": image_id,
            "stage": "stage4c_scale_best_candidate",
            "candidate_variant": "residual_floor_suppression",
            "gamma": args.gamma,
            "tau": args.tau,
            "baseline": "Stage 1 EditGuard-only watermarked/tampered inputs, recovered with original carrier.",
            "candidate": "EditGuard carrier with residual_floor_suppression before bitencoder.",
            "original_path": str(original_path.resolve()),
            "gt_mask_path": str(gt_mask_path.resolve()),
            "baseline_pred_mask_path": str((sample_out / "baseline_pred_mask.png").resolve()),
            "candidate_pred_mask_path": str((sample_out / "candidate_pred_mask.png").resolve()),
            "candidate_watermarked_path": str((sample_out / "candidate_watermarked.png").resolve()),
            "candidate_tampered_path": str((sample_out / "candidate_tampered.png").resolve()),
            "candidate_visualization_path": str(visual_path.resolve()),
            "message_seed": rng_seed,
            "notes": [
                "No training is performed.",
                "GT mask is used only as an oracle residual-floor suppression map and deterministic tamper region.",
                "GT mask is not used as the predicted mask.",
                "No Stable Diffusion, SAM, CLIP, or DINO module is used.",
            ],
        }
        (sample_out / "metrics.json").write_text(
            json.dumps(
                {
                    "image_id": image_id,
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "flat_row": row,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (sample_out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return row, visual_path
    except Exception as exc:
        sample_out.mkdir(parents=True, exist_ok=True)
        row = {field: "" for field in CSV_FIELDS}
        row.update(
            {
                "image_id": image_id,
                "status": "failed",
                "error_message": str(exc),
                "original_path": str((sample_dir / "original.png").resolve()),
                "gt_mask_path": str((sample_dir / "gt_mask.png").resolve()),
            }
        )
        (sample_out / "metrics.json").write_text(
            json.dumps({"image_id": image_id, "status": "failed", "error_message": str(exc), "flat_row": row}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return row, None


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def summarize(rows: list[dict[str, Any]], failed_cases: list[dict[str, str]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]

    def vals(prefix: str, metric_name: str) -> list[float]:
        key = f"{prefix}_{metric_name}"
        return [float(row[key]) for row in ok_rows if row.get(key) not in ("", None)]

    def delta_vals(metric_name: str) -> list[float]:
        key = f"delta_{metric_name}"
        return [float(row[key]) for row in ok_rows if row.get(key) not in ("", None)]

    payload: dict[str, Any] = {
        "total_samples": len(rows),
        "evaluated_samples": len(ok_rows),
        "failed_samples": len(failed_cases),
    }
    for metric_name in [
        "iou",
        "dice",
        "precision",
        "recall",
        "mae",
        "psnr",
        "ssim",
        "fragmentation_ratio",
        "small_pred_region_ratio",
        "tiny_pred_region_ratio",
        "semantic_compactness",
        "bit_accuracy",
        "ber",
    ]:
        baseline_values = vals("baseline", metric_name)
        candidate_values = vals("candidate", metric_name)
        payload[f"mean_{metric_name}_baseline"] = mean(baseline_values)
        payload[f"mean_{metric_name}_candidate"] = mean(candidate_values)
        payload[f"median_{metric_name}_baseline"] = median(baseline_values)
        payload[f"median_{metric_name}_candidate"] = median(candidate_values)
        if metric_name != "ber":
            payload[f"delta_mean_{metric_name}"] = mean(candidate_values) - mean(baseline_values)
            payload[f"delta_median_{metric_name}"] = median(candidate_values) - median(baseline_values)
    payload["mean_fragmentation_baseline"] = payload["mean_fragmentation_ratio_baseline"]
    payload["mean_fragmentation_candidate"] = payload["mean_fragmentation_ratio_candidate"]
    payload["delta_mean_fragmentation"] = payload["delta_mean_fragmentation_ratio"]
    payload["mean_small_region_ratio_baseline"] = payload["mean_small_pred_region_ratio_baseline"]
    payload["mean_small_region_ratio_candidate"] = payload["mean_small_pred_region_ratio_candidate"]
    payload["delta_mean_small_region_ratio"] = payload["delta_mean_small_pred_region_ratio"]
    payload["mean_tiny_region_ratio_baseline"] = payload["mean_tiny_pred_region_ratio_baseline"]
    payload["mean_tiny_region_ratio_candidate"] = payload["mean_tiny_pred_region_ratio_candidate"]
    payload["delta_mean_tiny_region_ratio"] = payload["delta_mean_tiny_pred_region_ratio"]
    payload["mean_ber_baseline"] = payload["mean_ber_baseline"]
    payload["mean_ber_candidate"] = payload["mean_ber_candidate"]
    payload["improved_iou_count"] = sum(bool(row.get("improved_iou")) for row in ok_rows)
    payload["improved_fragmentation_count"] = sum(bool(row.get("improved_fragmentation")) for row in ok_rows)
    payload["improved_both_count"] = sum(bool(row.get("improved_both")) for row in ok_rows)
    payload["worsened_iou_count"] = sum(bool(row.get("worsened_iou")) for row in ok_rows)
    payload["worsened_fragmentation_count"] = sum(bool(row.get("worsened_fragmentation")) for row in ok_rows)
    payload["stable_bit_accuracy_count"] = sum(bool(row.get("stable_bit_accuracy")) for row in ok_rows)
    payload["delta_iou_mean_of_per_sample_deltas"] = mean(delta_vals("iou"))
    payload["delta_fragmentation_mean_of_per_sample_deltas"] = mean(delta_vals("fragmentation_ratio"))
    return payload


def build_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        out.append({field: row.get(field, "") for field in DELTA_FIELDS})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_contact_sheet(vis_paths: list[Path], output_path: Path, max_samples: int) -> dict[str, Any]:
    existing = [path for path in vis_paths if path is not None and path.exists()]
    if len(existing) > max_samples:
        indices = sorted({round(i * (len(existing) - 1) / (max_samples - 1)) for i in range(max_samples)})
        selected = [existing[idx] for idx in indices]
        sampling = "even"
    else:
        selected = existing
        sampling = "all"

    rows = []
    for path in selected:
        rows.append(Image.open(path).convert("RGB").resize((882, 147), BICUBIC))
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
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = project_root / args.ckpt
    opt_path = project_root / args.opt
    if not input_dir.exists():
        raise FileNotFoundError(f"Stage 1 generated input directory not found: {input_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not opt_path.exists():
        raise FileNotFoundError(f"EditGuard option file not found: {opt_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current EditGuard model configuration.")

    sample_ids = load_sample_ids(project_root, input_dir, args.split, args.start, args.max_samples)
    model = load_editguard_model(project_root, opt_path, ckpt_path)
    rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, str]] = []
    vis_paths: list[Path] = []

    for sample_index, image_id in enumerate(sample_ids):
        row, visual_path = process_sample(model, image_id, sample_index + args.start, input_dir, output_dir, args)
        rows.append(row)
        if visual_path is not None:
            vis_paths.append(visual_path)
        if row.get("status") != "ok":
            failed_cases.append(
                {
                    "image_id": str(image_id),
                    "failure_stage": "stage4c_scale_process_sample",
                    "error_message": str(row.get("error_message", "")),
                }
            )
        if (sample_index + 1) % 50 == 0 or sample_index + 1 == len(sample_ids):
            ok_count = sum(row.get("status") == "ok" for row in rows)
            print(f"Progress: {sample_index + 1}/{len(sample_ids)} samples, ok={ok_count}, failed={len(failed_cases)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    per_sample_csv = output_dir / "stage4c_scale_per_sample_metrics.csv"
    write_csv(per_sample_csv, rows, CSV_FIELDS)
    delta_csv = output_dir / "stage4c_scale_delta_summary.csv"
    write_csv(delta_csv, build_delta_rows(rows), DELTA_FIELDS)
    failed_path = output_dir / "stage4c_scale_failed_cases.json"
    failed_path.write_text(json.dumps(failed_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_info = build_contact_sheet(vis_paths, output_dir / "stage4c_scale_contact_sheet.png", args.contact_sheet_samples)

    summary = summarize(rows, failed_cases)
    summary.update(
        {
            "stage": "stage4c_scale_best_candidate",
            "candidate_variant": "residual_floor_suppression",
            "gamma": args.gamma,
            "tau": args.tau,
            "oracle_note": "GT mask is used only as oracle residual-floor suppression map; it is not used as predicted mask.",
            "stability_note": (
                "Candidate improved both IoU and fragmentation for a subset of samples; inspect count statistics for stability."
            ),
            "contact_sheet": contact_info,
        }
    )
    if summary["delta_mean_iou"] <= 0.0 or summary["delta_mean_fragmentation"] >= 0.0:
        summary["stability_note"] = (
            "Candidate improvement is not stable at 1000-sample scale; do not overstate the 20-sample Pareto result."
        )
    summary_path = output_dir / "stage4c_scale_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved per-sample metrics: {per_sample_csv}")
    print(f"Saved delta summary: {delta_csv}")
    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved failed cases: {failed_path}")
    print(f"Saved contact sheet: {output_dir / 'stage4c_scale_contact_sheet.png'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
