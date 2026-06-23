from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

CASE_FIELDS = [
    "image_id",
    "asem_mode",
    "modulation_variant",
    "gamma",
    "tau",
    "smooth_sigma",
    "case_key",
    "output_dir",
    "watermarked_path",
    "tampered_path",
    "pred_mask_path",
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
    "copyright_eval_available",
    "bit_accuracy",
    "ber",
    "status",
    "error_message",
]

GROUP_KEYS = ["asem_mode", "modulation_variant", "gamma", "tau", "smooth_sigma", "case_key"]
GROUP_FIELDS = GROUP_KEYS + [
    "evaluated_samples",
    "mean_psnr",
    "mean_ssim",
    "mean_iou",
    "mean_dice",
    "mean_precision",
    "mean_recall",
    "mean_mae",
    "mean_num_pred_regions",
    "mean_fragmentation_ratio",
    "mean_small_pred_region_ratio",
    "mean_tiny_pred_region_ratio",
    "mean_semantic_compactness",
    "mean_bit_accuracy",
    "mean_ber",
]

PARETO_FIELDS = GROUP_FIELDS + [
    "delta_iou_vs_baseline",
    "delta_dice_vs_baseline",
    "delta_psnr_vs_baseline",
    "delta_fragmentation_ratio_vs_baseline",
    "delta_small_pred_region_ratio_vs_baseline",
    "delta_tiny_pred_region_ratio_vs_baseline",
    "pareto_reason",
]


@dataclass(frozen=True)
class SuppressionCase:
    variant: str
    gamma: float
    tau: float
    smooth_sigma: float
    is_baseline: bool = False

    @property
    def key(self) -> str:
        parts = [self.variant, f"gamma_{format_number(self.gamma)}"]
        if self.tau > 0.0:
            parts.append(f"tau_{format_number(self.tau)}")
        if self.smooth_sigma > 0.0:
            parts.append(f"sigma_{format_number(self.smooth_sigma)}")
        return "__".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4C fragile residual suppression/reallocation ablation without training."
    )
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--input_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage4c_residual_suppression")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--asem_mode", default="gt_mask_boost", choices=["gt_mask_boost"])
    parser.add_argument("--gamma", nargs="*", type=float, default=[0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--tau", nargs="*", type=float, default=[0.01, 0.02, 0.03])
    parser.add_argument("--smooth_sigma", nargs="*", type=float, default=[3.0, 5.0])
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
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


def sample_dirs(input_dir: Path, start: int, max_samples: int) -> list[Path]:
    dirs = sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit())
    return dirs[start : start + max_samples]


def format_number(value: float) -> str:
    return f"{value:.4g}".replace("-", "m").replace(".", "p")


def build_cases(args: argparse.Namespace) -> list[SuppressionCase]:
    cases = [SuppressionCase("baseline", 0.0, 0.0, 0.0, is_baseline=True)]
    for gamma in args.gamma:
        cases.append(SuppressionCase("background_suppression", gamma, 0.0, 0.0))
        cases.append(SuppressionCase("area_reallocation", gamma, 0.0, 0.0))
        for sigma in args.smooth_sigma:
            cases.append(SuppressionCase("soft_background_suppression", gamma, 0.0, sigma))
        for tau in args.tau:
            cases.append(SuppressionCase("residual_floor_suppression", gamma, tau, 0.0))
    return cases


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


def build_asem_map(mode: str, gt_mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    if mode != "gt_mask_boost":
        raise ValueError(f"Stage 4C only supports gt_mask_boost oracle suppression map, got: {mode}")
    mask_img = Image.fromarray(gt_mask.astype(np.uint8) * 255, mode="L")
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), NEAREST)
    return (np.asarray(mask_img, dtype=np.float32) / 255.0).clip(0.0, 1.0)


def effective_visual_map(asem_map: np.ndarray, case: SuppressionCase) -> np.ndarray:
    if case.is_baseline:
        return np.zeros_like(asem_map, dtype=np.float32)
    if case.variant == "soft_background_suppression" and case.smooth_sigma > 0.0:
        return cv2.GaussianBlur(asem_map, (0, 0), sigmaX=case.smooth_sigma, sigmaY=case.smooth_sigma).clip(0.0, 1.0)
    return asem_map


def error_map_image(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[np.logical_and(pred, gt)] = [255, 255, 255]
    out[np.logical_and(pred, ~gt)] = [255, 80, 80]
    out[np.logical_and(~pred, gt)] = [80, 160, 255]
    return Image.fromarray(out, mode="RGB")


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


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
    watermarked: Image.Image,
    tampered: Image.Image,
    gt: np.ndarray,
    pred: np.ndarray,
    asem_map: np.ndarray,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    asem_image = Image.fromarray(np.clip(asem_map * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    panels = [
        title_panel(original, "original"),
        title_panel(asem_image, "oracle map"),
        title_panel(watermarked, f"PSNR {float(row['psnr']):.2f}"),
        title_panel(tampered, "tampered"),
        title_panel(mask_image(gt), "gt_mask"),
        title_panel(mask_image(pred), f"IoU {float(row['iou']):.3f}"),
        title_panel(error_map_image(gt, pred), f"frag {float(row['fragmentation_ratio']):.1f}"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


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


def output_complete(sample_out: Path) -> bool:
    required = ["watermarked.png", "tampered.png", "pred_mask.png", "carrier_mod_vis.png", "metrics.json", "meta.json"]
    return all((sample_out / name).exists() for name in required)


def load_existing_metrics(sample_out: Path) -> dict[str, Any]:
    row = json.loads((sample_out / "metrics.json").read_text(encoding="utf-8"))
    row["output_dir"] = str(sample_out.resolve())
    return {field: row.get(field, "") for field in CASE_FIELDS}


def process_case(
    model: Any,
    sample_dir: Path,
    sample_index: int,
    case: SuppressionCase,
    project_root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path | None]:
    image_id = sample_dir.name
    sample_out = project_root / args.output_dir / case.key / image_id
    visual_path = sample_out / "carrier_mod_vis.png"
    if output_complete(sample_out) and not args.overwrite:
        return load_existing_metrics(sample_out), visual_path

    row: dict[str, Any] = {field: "" for field in CASE_FIELDS}
    row.update(
        {
            "image_id": image_id,
            "asem_mode": args.asem_mode,
            "modulation_variant": case.variant,
            "gamma": case.gamma,
            "tau": case.tau,
            "smooth_sigma": case.smooth_sigma,
            "case_key": case.key,
            "output_dir": str(sample_out.resolve()),
            "watermarked_path": str((sample_out / "watermarked.png").resolve()),
            "tampered_path": str((sample_out / "tampered.png").resolve()),
            "pred_mask_path": str((sample_out / "pred_mask.png").resolve()),
            "status": "ok",
            "error_message": "",
        }
    )

    try:
        original_path = sample_dir / "original.png"
        gt_mask_path = sample_dir / "gt_mask.png"
        for required in [original_path, gt_mask_path]:
            if not required.exists():
                raise FileNotFoundError(f"Missing required input: {required}")

        sample_out.mkdir(parents=True, exist_ok=True)
        original = Image.open(original_path).convert("RGB").resize((512, 512), BICUBIC)
        gt_img = Image.open(gt_mask_path).convert("L").resize((512, 512), NEAREST)
        original_arr = np.asarray(original, dtype=np.uint8)
        gt = np.asarray(gt_img, dtype=np.uint8) > args.mask_threshold
        if int(gt.sum()) == 0:
            raise ValueError(f"GT mask is empty: {gt_mask_path}")

        asem_map = build_asem_map(args.asem_mode, gt, original_arr.shape[:2])
        asem_tensor = torch.from_numpy(asem_map).float().unsqueeze(0).unsqueeze(0).to(model.device)
        rng_seed = args.seed + int(image_id) if image_id.isdigit() else args.seed + sample_index
        rng = np.random.default_rng(rng_seed)
        message_bits = rng.choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)

        data = image_to_editguard_batch(original_arr)
        data["MES"] = message_bits
        model.feed_data(data)
        model.mes = message_bits
        watermarked = model.image_hiding(
            asem_map=(None if case.is_baseline else asem_tensor),
            asem_alpha=case.gamma,
            asem_mode=("none" if case.is_baseline else args.asem_mode),
            asem_variant=("naive_boost" if case.is_baseline else case.variant),
            asem_tau=case.tau,
            asem_smooth_sigma=case.smooth_sigma,
        )
        Image.fromarray(watermarked).save(sample_out / "watermarked.png")

        tampered = deterministic_tamper(watermarked, gt, sample_index)
        Image.fromarray(tampered).save(sample_out / "tampered.png")

        tampered_data = image_to_editguard_batch(tampered)
        tampered_data["MES"] = None
        model.feed_data(tampered_data)
        pred_mask, recmessage = model.image_recovery(args.recovery_threshold)
        pred = np.asarray(pred_mask) > 0
        Image.fromarray(pred.astype(np.uint8) * 255, mode="L").save(sample_out / "pred_mask.png")

        metrics: dict[str, Any] = {}
        metrics.update(pixel_metrics(gt, pred))
        metrics.update(component_metrics(gt, pred, args.small_area_pixels, args.tiny_area_pixels))
        metrics["psnr"] = psnr(original_arr, watermarked)
        metrics["ssim"] = ssim_rgb(original_arr, watermarked)
        metrics.update(evaluate_copyright(message_bits, recmessage))

        row.update({key: metrics.get(key, row.get(key, "")) for key in CASE_FIELDS})
        row.update(
            {
                "image_id": image_id,
                "asem_mode": args.asem_mode,
                "modulation_variant": case.variant,
                "gamma": case.gamma,
                "tau": case.tau,
                "smooth_sigma": case.smooth_sigma,
                "case_key": case.key,
                "output_dir": str(sample_out.resolve()),
                "watermarked_path": str((sample_out / "watermarked.png").resolve()),
                "tampered_path": str((sample_out / "tampered.png").resolve()),
                "pred_mask_path": str((sample_out / "pred_mask.png").resolve()),
                "status": "ok",
                "error_message": "",
            }
        )

        save_visualization(
            original,
            Image.fromarray(watermarked),
            Image.fromarray(tampered),
            gt,
            pred,
            effective_visual_map(asem_map, case),
            row,
            visual_path,
        )

        formula = {
            "baseline": "out_y_mod = out_y",
            "background_suppression": "residual_mod = residual * (1 - gamma * (1 - A))",
            "soft_background_suppression": "A_smooth = gaussian_blur(A); residual_mod = residual * (1 - gamma * (1 - A_smooth))",
            "residual_floor_suppression": "small outside-GT residual values below tau are multiplied by (1 - gamma)",
            "area_reallocation": "background_suppression followed by mean-absolute-energy normalization",
        }[case.variant]
        meta = {
            "image_id": image_id,
            "stage": "stage4c_residual_suppression",
            "asem_mode": args.asem_mode,
            "modulation_variant": case.variant,
            "gamma": case.gamma,
            "tau": case.tau,
            "smooth_sigma": case.smooth_sigma,
            "formula": formula,
            "original_path": str(original_path.resolve()),
            "gt_mask_path": str(gt_mask_path.resolve()),
            "watermarked_path": str((sample_out / "watermarked.png").resolve()),
            "tampered_path": str((sample_out / "tampered.png").resolve()),
            "pred_mask_path": str((sample_out / "pred_mask.png").resolve()),
            "visualization_path": str(visual_path.resolve()),
            "recovery_threshold": args.recovery_threshold,
            "mask_threshold": args.mask_threshold,
            "message_seed": rng_seed,
            "notes": [
                "This is a no-training ablation.",
                "GT mask is used only as the oracle suppression/reallocation map and deterministic tamper region.",
                "GT mask is not used as the predicted mask.",
                "No Stable Diffusion, SAM, CLIP, or DINO module is used.",
                "The robust bitencoder/bitdecoder architecture is unchanged.",
            ],
        }
        (sample_out / "metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        (sample_out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return row, visual_path
    except Exception as exc:
        row.update({"status": "failed", "error_message": str(exc)})
        sample_out.mkdir(parents=True, exist_ok=True)
        (sample_out / "metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        return row, None


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        "psnr",
        "ssim",
        "iou",
        "dice",
        "precision",
        "recall",
        "mae",
        "num_pred_regions",
        "fragmentation_ratio",
        "small_pred_region_ratio",
        "tiny_pred_region_ratio",
        "semantic_compactness",
        "bit_accuracy",
        "ber",
    ]
    groups: dict[tuple[str, str, float, float, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[
                (
                    str(row["asem_mode"]),
                    str(row["modulation_variant"]),
                    float(row["gamma"]),
                    float(row["tau"]),
                    float(row["smooth_sigma"]),
                    str(row["case_key"]),
                )
            ].append(row)

    summaries = []
    for key, group_rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][3], item[0][4])):
        mode, variant, gamma, tau, sigma, case_key = key
        item: dict[str, Any] = {
            "asem_mode": mode,
            "modulation_variant": variant,
            "gamma": gamma,
            "tau": tau,
            "smooth_sigma": sigma,
            "case_key": case_key,
            "evaluated_samples": len(group_rows),
        }
        for field in numeric_fields:
            vals = [float(row[field]) for row in group_rows if row.get(field) not in ("", None)]
            item[f"mean_{field}"] = mean(vals)
        summaries.append(item)
    return summaries


def find_baseline(groups: list[dict[str, Any]]) -> dict[str, Any]:
    for row in groups:
        if row["modulation_variant"] == "baseline":
            return row
    raise RuntimeError("Baseline group is missing; cannot compute Pareto candidates.")


def pareto_candidates(groups: list[dict[str, Any]], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in groups:
        if row["modulation_variant"] == "baseline":
            continue
        iou_ok = float(row["mean_iou"]) >= float(baseline["mean_iou"]) - 0.02
        dice_ok = float(row["mean_dice"]) >= float(baseline["mean_dice"]) - 0.02
        psnr_ok = float(row["mean_psnr"]) >= float(baseline["mean_psnr"]) - 1.0
        bit_ok = float(row["mean_bit_accuracy"]) >= 0.99
        frag_ok = float(row["mean_fragmentation_ratio"]) < float(baseline["mean_fragmentation_ratio"])
        small_ok = float(row["mean_small_pred_region_ratio"]) < float(baseline["mean_small_pred_region_ratio"])
        tiny_ok = float(row["mean_tiny_pred_region_ratio"]) < float(baseline["mean_tiny_pred_region_ratio"])
        if iou_ok and dice_ok and psnr_ok and bit_ok and frag_ok and (small_ok or tiny_ok):
            item = dict(row)
            item["delta_iou_vs_baseline"] = float(row["mean_iou"]) - float(baseline["mean_iou"])
            item["delta_dice_vs_baseline"] = float(row["mean_dice"]) - float(baseline["mean_dice"])
            item["delta_psnr_vs_baseline"] = float(row["mean_psnr"]) - float(baseline["mean_psnr"])
            item["delta_fragmentation_ratio_vs_baseline"] = (
                float(row["mean_fragmentation_ratio"]) - float(baseline["mean_fragmentation_ratio"])
            )
            item["delta_small_pred_region_ratio_vs_baseline"] = (
                float(row["mean_small_pred_region_ratio"]) - float(baseline["mean_small_pred_region_ratio"])
            )
            item["delta_tiny_pred_region_ratio_vs_baseline"] = (
                float(row["mean_tiny_pred_region_ratio"]) - float(baseline["mean_tiny_pred_region_ratio"])
            )
            reasons = ["lower_fragmentation_ratio"]
            if small_ok:
                reasons.append("lower_small_pred_region_ratio")
            if tiny_ok:
                reasons.append("lower_tiny_pred_region_ratio")
            item["pareto_reason"] = "+".join(reasons)
            out.append(item)
    return sorted(
        out,
        key=lambda row: (
            float(row["delta_fragmentation_ratio_vs_baseline"]),
            float(row["delta_small_pred_region_ratio_vs_baseline"]),
            float(row["delta_tiny_pred_region_ratio_vs_baseline"]),
            -float(row["mean_iou"]),
        ),
    )


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

    selected_samples = sample_dirs(input_dir, args.start, args.max_samples)
    cases = build_cases(args)
    model = load_editguard_model(project_root, opt_path, ckpt_path)
    rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, str]] = []
    vis_paths: list[Path] = []

    total_cases = len(selected_samples) * len(cases)
    case_idx = 0
    for sample_index, sample_dir in enumerate(selected_samples):
        for case in cases:
            case_idx += 1
            row, visual_path = process_case(model, sample_dir, sample_index, case, project_root, args)
            rows.append(row)
            if visual_path is not None:
                vis_paths.append(visual_path)
            if row.get("status") != "ok":
                failed_cases.append(
                    {
                        "image_id": str(row.get("image_id", sample_dir.name)),
                        "case_key": str(case.key),
                        "failure_stage": "stage4c_process_case",
                        "error_message": str(row.get("error_message", "")),
                    }
                )
            if case_idx % 50 == 0 or case_idx == total_cases:
                ok_count = sum(row.get("status") == "ok" for row in rows)
                print(f"Progress: {case_idx}/{total_cases} cases, ok={ok_count}, failed={len(failed_cases)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    case_csv = output_dir / "stage4c_case_metrics.csv"
    write_csv(case_csv, rows, CASE_FIELDS)
    group_rows = group_summary(rows)
    group_csv = output_dir / "stage4c_group_summary.csv"
    write_csv(group_csv, group_rows, GROUP_FIELDS)
    baseline = find_baseline(group_rows)
    pareto_rows = pareto_candidates(group_rows, baseline)
    pareto_csv = output_dir / "stage4c_pareto_candidates.csv"
    write_csv(pareto_csv, pareto_rows, PARETO_FIELDS)
    failed_path = output_dir / "failed_cases.json"
    failed_path.write_text(json.dumps(failed_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_info = build_contact_sheet(vis_paths, output_dir / "stage4c_contact_sheet.png", args.contact_sheet_samples)

    negative_result = len(pareto_rows) == 0
    summary_payload = {
        "stage": "stage4c_residual_suppression",
        "total_cases": len(rows),
        "evaluated_cases": sum(row.get("status") == "ok" for row in rows),
        "failed_cases": len(failed_cases),
        "max_samples": args.max_samples,
        "asem_mode": args.asem_mode,
        "gamma": args.gamma,
        "tau": args.tau,
        "smooth_sigma": args.smooth_sigma,
        "baseline": baseline,
        "pareto_candidate_count": len(pareto_rows),
        "negative_result": negative_result,
        "negative_result_note": (
            "No residual suppression/reallocation variant satisfied the Pareto constraints; "
            "inference-time residual modulation is insufficient and training-time carrier regularization is needed."
            if negative_result
            else "At least one residual suppression/reallocation variant satisfied the Pareto constraints."
        ),
        "notes": [
            "This ablation does not train EditGuard.",
            "gt_mask is used only as an oracle suppression/reallocation map for mechanism validation.",
            "GT mask is not used as the predicted mask.",
            "No Stable Diffusion, SAM, CLIP, or DINO module is used.",
            "The robust bitencoder/bitdecoder architecture is unchanged.",
        ],
        "contact_sheet": contact_info,
    }
    (output_dir / "stage4c_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved case metrics: {case_csv}")
    print(f"Saved group summary: {group_csv}")
    print(f"Saved Pareto candidates: {pareto_csv}")
    print(f"Saved summary JSON: {output_dir / 'stage4c_summary.json'}")
    print(f"Saved failed cases: {failed_path}")
    print(f"Saved contact sheet: {output_dir / 'stage4c_contact_sheet.png'}")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
