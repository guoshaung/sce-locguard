from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

TRAIN_LOG_FIELDS = [
    "group",
    "step",
    "image_id",
    "original_loss",
    "l_forw_fit",
    "l_back_rec",
    "l_center_x",
    "l_msg",
    "carrier_sparsity_loss",
    "background_residual_suppression_loss",
    "residual_floor_loss",
    "semantic_guided_residual_loss",
    "carrier_regularization_total",
    "total_loss",
    "learning_rate",
    "trainable_parameter_count",
    "frozen_parameter_count",
    "status",
    "error_message",
]

EVAL_FIELDS = [
    "group",
    "image_id",
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


@dataclass(frozen=True)
class ExperimentGroup:
    name: str
    enable_carrier_regularization: bool
    lambda_sparse: float
    lambda_bg: float
    lambda_floor: float
    lambda_sem: float
    residual_floor_tau: float


GROUPS = [
    ExperimentGroup("baseline_finetune", False, 0.0, 0.0, 0.0, 0.0, 0.01),
    ExperimentGroup("floor_reg_weak", True, 0.0, 0.0, 0.01, 0.0, 0.01),
    ExperimentGroup("bg_floor_reg_weak", True, 0.0, 0.005, 0.01, 0.0, 0.01),
    ExperimentGroup("floor_reg_medium", True, 0.0, 0.0, 0.05, 0.0, 0.01),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6A small-scale fine-tuning sanity check.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--stage1_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage6a_finetune_sanity")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--max_train_samples", type=int, default=20)
    parser.add_argument("--max_val_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional subset of group names to run.")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--recovery_threshold", type=float, default=0.2)
    parser.add_argument("--small_area_pixels", type=int, default=100)
    parser.add_argument("--tiny_area_pixels", type=int, default=25)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_editguard_imports(project_root: Path) -> None:
    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def load_train_model(project_root: Path, opt_path: Path, ckpt_path: Path, group: ExperimentGroup, lr: float):
    configure_editguard_imports(project_root)
    import options.options as option
    from models import create_model as create_model_editguard

    logging.getLogger("base").setLevel(logging.ERROR)
    opt = option.parse(str(opt_path), is_train=True)
    # test_editguard.yml omits this train-time field; keep message loss logging usable.
    opt["losstype"] = opt.get("losstype") or "mse"
    opt["sdinpaint"] = False
    opt["controlnetinpaint"] = False
    opt["sdxl"] = False
    opt["repaint"] = False
    opt["dist"] = False
    opt["gpu_ids"] = [0]
    opt["path"]["pretrain_model_G"] = str(ckpt_path)
    opt["path"]["strict_load"] = True
    opt["train"]["lr_G"] = lr
    opt["train"]["enable_carrier_regularization"] = group.enable_carrier_regularization
    opt["train"]["lambda_sparse"] = group.lambda_sparse
    opt["train"]["lambda_bg"] = group.lambda_bg
    opt["train"]["lambda_floor"] = group.lambda_floor
    opt["train"]["lambda_sem"] = group.lambda_sem
    opt["train"]["residual_floor_tau"] = group.residual_floor_tau
    opt["train"]["return_fragile_carrier"] = True
    opt["train"]["freeze_bitencoder"] = True
    opt["train"]["freeze_bitdecoder"] = True
    opt["train"]["freeze_copyright_branch"] = True
    opt = option.dict_to_nonedict(opt)
    model = create_model_editguard(opt)
    return model


def scalar_value(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def sample_dirs(stage1_dir: Path, max_samples: int) -> list[Path]:
    return sorted(path for path in stage1_dir.iterdir() if path.is_dir() and path.name.isdigit())[:max_samples]


def image_to_editguard_batch(image: np.ndarray, gt_mask: np.ndarray | None = None) -> dict[str, torch.Tensor]:
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
    data = {"LQ": list_h, "GT": img_gt}
    if gt_mask is not None:
        data["GT_MASK"] = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return data


def load_original_and_mask(sample_dir: Path, threshold: int) -> tuple[Image.Image, np.ndarray]:
    original = Image.open(sample_dir / "original.png").convert("RGB").resize((512, 512), BICUBIC)
    gt_mask = np.asarray(
        Image.open(sample_dir / "gt_mask.png").convert("L").resize((512, 512), NEAREST),
        dtype=np.uint8,
    ) > threshold
    return original, gt_mask


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


def message_bits_for(image_id: str, seed: int, offset: int = 0) -> np.ndarray:
    rng_seed = seed + int(image_id) + offset if image_id.isdigit() else seed + offset
    rng = np.random.default_rng(rng_seed)
    return rng.choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)


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


def evaluate_copyright(message_bits: np.ndarray, recmessage: torch.Tensor) -> dict[str, Any]:
    try:
        rec = recmessage.detach().float().cpu().numpy().reshape(-1)
        truth = (message_bits.reshape(-1) > 0).astype(np.uint8)
        pred = (rec > 0.5).astype(np.uint8)
        if pred.shape[0] != truth.shape[0]:
            raise ValueError(f"Recovered bit length mismatch: expected {truth.shape[0]}, got {pred.shape[0]}")
        bit_accuracy = float(np.mean(pred == truth))
        return {"copyright_eval_available": True, "bit_accuracy": bit_accuracy, "ber": float(1.0 - bit_accuracy)}
    except Exception as exc:
        return {"copyright_eval_available": False, "bit_accuracy": "", "ber": "", "copyright_error_message": str(exc)}


def count_parameters(model: Any) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.netG.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.netG.parameters() if not param.requires_grad)
    return int(trainable), int(frozen)


def module_parameter_lines(model: Any, trainable: bool) -> list[str]:
    totals: dict[str, int] = {}
    for name, param in model.netG.named_parameters():
        if bool(param.requires_grad) == trainable:
            module_name = ".".join(name.split(".")[:2])
            totals[module_name] = totals.get(module_name, 0) + int(param.numel())
    return [f"{name}: {count}" for name, count in sorted(totals.items())]


def set_train_mode_for_finetune(model: Any) -> None:
    module = model.netG.module if hasattr(model.netG, "module") else model.netG
    model.netG.train()
    module.bitencoder.eval()
    module.bitdecoder.eval()


def train_step(model: Any, sample_dir: Path, step: int, group: ExperimentGroup, args: argparse.Namespace) -> dict[str, Any]:
    from models.IBSN import dwt, iwt

    image_id = sample_dir.name
    original, gt_mask = load_original_and_mask(sample_dir, args.mask_threshold)
    data = image_to_editguard_batch(np.asarray(original), gt_mask)
    model.feed_data(data)
    set_train_mode_for_finetune(model)
    model.optimizer_G.zero_grad(set_to_none=True)

    b, n, t, c, h, w = model.ref_L.shape
    center = t // 2
    intval = model.gop // 2
    host = model.real_H[:, center - intval:center + intval + 1]
    secret = model.ref_L[:, :, center - intval:center + intval + 1]
    message_np = message_bits_for(image_id, args.seed, offset=step)
    message = torch.from_numpy(message_np).to(model.device)

    forw_result = model.netG(
        x=dwt(host.reshape(b, -1, h, w)),
        x_h=dwt(secret[:, 0].reshape(b, -1, h, w)),
        message=message,
        return_fragile_carrier=True,
    )
    container = forw_result["encoded_image"]
    y_forw = container
    l_forw_fit = model.loss_forward(y_forw, host[:, 0])
    y = model.Quantization(y_forw)
    all_zero = torch.zeros(message.shape).to(model.device)
    out_x, out_x_h, _out_z, recmessage = model.netG(x=y, message=all_zero, rev=True)
    out_x = iwt(out_x)
    out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
    l_back_rec = model.loss_back_rec(out_x, host[:, 0])
    out_x_h = torch.stack(out_x_h, dim=1)
    l_center_x = model.loss_back_rec(out_x_h[:, 0], secret[:, 0].reshape(b, -1, h, w))
    recmessage = torch.clamp(recmessage, -0.5, 0.5)
    l_msg = model.Reconstruction_msg(message, recmessage)
    original_loss = l_forw_fit * 2 + l_back_rec + l_center_x * 4
    carrier_losses = model.compute_carrier_regularization_losses(
        forw_result["residual"],
        gt_mask=model.gt_mask,
        semantic_map=None,
    )
    carrier_total = carrier_losses["carrier_regularization_total"] if group.enable_carrier_regularization else original_loss * 0.0
    total_loss = original_loss + carrier_total
    if not torch.isfinite(total_loss):
        raise FloatingPointError(f"Non-finite total loss at step {step}: {float(total_loss.detach().cpu())}")
    total_loss.backward()
    if model.train_opt["gradient_clipping"]:
        nn.utils.clip_grad_norm_(model.netG.parameters(), model.train_opt["gradient_clipping"])
    model.optimizer_G.step()

    trainable_count, frozen_count = count_parameters(model)
    return {
        "group": group.name,
        "step": step,
        "image_id": image_id,
        "original_loss": scalar_value(original_loss),
        "l_forw_fit": scalar_value(l_forw_fit),
        "l_back_rec": scalar_value(l_back_rec),
        "l_center_x": scalar_value(l_center_x),
        "l_msg": scalar_value(l_msg),
        "carrier_sparsity_loss": scalar_value(carrier_losses["carrier_sparsity_loss"]),
        "background_residual_suppression_loss": scalar_value(carrier_losses["background_residual_suppression_loss"]),
        "residual_floor_loss": scalar_value(carrier_losses["residual_floor_loss"]),
        "semantic_guided_residual_loss": scalar_value(carrier_losses["semantic_guided_residual_loss"]),
        "carrier_regularization_total": scalar_value(carrier_total),
        "total_loss": scalar_value(total_loss),
        "learning_rate": float(model.optimizer_G.param_groups[0]["lr"]),
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": frozen_count,
        "status": "ok",
        "error_message": "",
    }


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


def save_eval_visual(
    original: Image.Image,
    watermarked: Image.Image,
    tampered: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    row: dict[str, Any],
    output_path: Path,
) -> None:
    panels = [
        title_panel(original, f"{row['group']} original"),
        title_panel(watermarked, f"PSNR {float(row['psnr']):.2f}"),
        title_panel(tampered, "tampered"),
        title_panel(mask_image(gt_mask), "gt_mask"),
        title_panel(mask_image(pred_mask), f"IoU {float(row['iou']):.3f}"),
        title_panel(error_map_image(gt_mask, pred_mask), f"frag {float(row['fragmentation_ratio']):.1f}"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def evaluate_group(model: Any, group: ExperimentGroup, val_samples: list[Path], output_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    vis_paths: list[Path] = []
    model.netG.eval()
    with torch.no_grad():
        for sample_index, sample_dir in enumerate(val_samples):
            image_id = sample_dir.name
            row = {field: "" for field in EVAL_FIELDS}
            row.update({"group": group.name, "image_id": image_id, "status": "ok", "error_message": ""})
            try:
                original, gt_mask = load_original_and_mask(sample_dir, args.mask_threshold)
                original_arr = np.asarray(original, dtype=np.uint8)
                message_np = message_bits_for(image_id, args.seed, offset=10000)
                data = image_to_editguard_batch(original_arr, gt_mask)
                data["MES"] = message_np
                model.feed_data(data)
                model.mes = message_np
                watermarked = model.image_hiding()
                tampered = deterministic_tamper(watermarked, gt_mask, sample_index)
                recovery_data = image_to_editguard_batch(tampered, gt_mask)
                recovery_data["MES"] = None
                model.feed_data(recovery_data)
                pred_mask, recmessage = model.image_recovery(args.recovery_threshold)
                pred = np.asarray(pred_mask) > 0

                row.update(
                    {
                        "psnr": psnr(original_arr, watermarked),
                        "ssim": ssim_rgb(original_arr, watermarked),
                    }
                )
                row.update(pixel_metrics(gt_mask, pred))
                row.update(component_metrics(gt_mask, pred, args.small_area_pixels, args.tiny_area_pixels))
                row.update(evaluate_copyright(message_np, recmessage))
                vis_path = output_dir / "visualizations" / group.name / f"{image_id}_eval.png"
                save_eval_visual(original, Image.fromarray(watermarked), Image.fromarray(tampered), gt_mask, pred, row, vis_path)
                vis_paths.append(vis_path)
            except Exception as exc:
                row.update({"status": "failed", "error_message": str(exc)})
            rows.append(row)
    return rows, vis_paths


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def summarize_eval(
    eval_rows: list[dict[str, Any]],
    failed_cases: list[dict[str, str]],
    groups: list[ExperimentGroup],
) -> dict[str, Any]:
    group_payload: dict[str, Any] = {}
    for group in groups:
        rows = [row for row in eval_rows if row["group"] == group.name and row["status"] == "ok"]
        payload: dict[str, Any] = {"evaluated_samples": len(rows)}
        for metric_name in [
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
        ]:
            vals = [float(row[metric_name]) for row in rows if row.get(metric_name) not in ("", None)]
            payload[f"mean_{metric_name}"] = mean(vals)
            payload[f"median_{metric_name}"] = median(vals)
        group_payload[group.name] = payload
    return {
        "stage": "stage6a_finetune_sanity",
        "groups": group_payload,
        "failed_cases": len(failed_cases),
        "sanity_note": "This uses valAGE-Set samples as a temporary technical sanity dataset, not a formal training result.",
        "no_external_semantic_modules": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    should_write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if should_write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def write_partial_status(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "stage6a_partial_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_contact_sheet(vis_paths: list[Path], output_path: Path, max_samples: int) -> dict[str, Any]:
    existing = [path for path in vis_paths if path.exists()]
    if len(existing) > max_samples:
        indices = sorted({round(i * (len(existing) - 1) / (max_samples - 1)) for i in range(max_samples)})
        selected = [existing[idx] for idx in indices]
        sampling = "even"
    else:
        selected = existing
        sampling = "all"
    rows = [Image.open(path).convert("RGB").resize((900, 175), BICUBIC) for path in selected]
    if not rows:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}
    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(rows)}


def save_checkpoint(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = model.netG.module if hasattr(model.netG, "module") else model.netG
    state_dict = {key: value.detach().cpu() for key, value in module.state_dict().items()}
    torch.save(state_dict, path)


def append_module_report(path: Path, group_name: str, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{group_name}]\n")
        for line in lines:
            f.write(f"{line}\n")
        f.write("\n")


def collapse_detected(eval_rows: list[dict[str, Any]], group_name: str) -> str | None:
    rows = [row for row in eval_rows if row["group"] == group_name and row["status"] == "ok"]
    if not rows:
        return "no successful validation samples"
    mean_bit = mean([float(row["bit_accuracy"]) for row in rows if row.get("bit_accuracy") not in ("", None)])
    mean_psnr = mean([float(row["psnr"]) for row in rows])
    mean_iou = mean([float(row["iou"]) for row in rows])
    mean_dice = mean([float(row["dice"]) for row in rows])
    if mean_bit < 0.95:
        return f"bit accuracy collapsed: {mean_bit:.4f}"
    if mean_psnr < 30.0:
        return f"PSNR dropped below sanity threshold: {mean_psnr:.4f}"
    if mean_iou < 0.2 or mean_dice < 0.3:
        return f"localization collapsed: IoU={mean_iou:.4f}, Dice={mean_dice:.4f}"
    return None


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Stage 6A sanity script currently supports batch_size=1 only.")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    project_root = Path(args.project_root).resolve()
    stage1_dir = (project_root / args.stage1_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_samples = sample_dirs(stage1_dir, args.max_train_samples)
    val_samples = sample_dirs(stage1_dir, args.max_val_samples)
    if not train_samples or not val_samples:
        raise FileNotFoundError(f"No Stage 1 sample directories found under {stage1_dir}")

    train_log_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, str]] = []
    all_vis_paths: list[Path] = []
    train_log_path = output_dir / "stage6a_training_log.csv"
    eval_metrics_path = output_dir / "stage6a_eval_metrics.csv"
    write_csv(train_log_path, [], TRAIN_LOG_FIELDS)
    write_csv(eval_metrics_path, [], EVAL_FIELDS)
    trainable_report = output_dir / "trainable_modules.txt"
    frozen_report = output_dir / "frozen_modules.txt"
    trainable_report.write_text(
        "Stage 6A trainable modules. Dataset note: valAGE-Set is used only as a temporary technical sanity dataset, not a formal training result.\n\n",
        encoding="utf-8",
    )
    frozen_report.write_text(
        "Stage 6A frozen modules. Copyright branch freeze is requested for all groups.\n\n",
        encoding="utf-8",
    )

    selected_groups = GROUPS
    if args.groups:
        group_lookup = {group.name: group for group in GROUPS}
        unknown = [name for name in args.groups if name not in group_lookup]
        if unknown:
            raise ValueError(f"Unknown Stage 6A group(s): {unknown}. Available: {sorted(group_lookup)}")
        selected_groups = [group_lookup[name] for name in args.groups]

    for group in selected_groups:
        model = None
        group_failed = False
        try:
            print(f"[Stage6A] Starting group {group.name}: max_steps={args.max_steps}, train_samples={len(train_samples)}, val_samples={len(val_samples)}", flush=True)
            model = load_train_model(project_root, project_root / args.opt, project_root / args.ckpt, group, args.lr)
            trainable_count, frozen_count = count_parameters(model)
            append_module_report(trainable_report, group.name, [f"total_trainable_parameters: {trainable_count}"] + module_parameter_lines(model, True))
            append_module_report(frozen_report, group.name, [f"total_frozen_parameters: {frozen_count}"] + module_parameter_lines(model, False))

            for step in range(1, args.max_steps + 1):
                sample_dir = train_samples[(step - 1) % len(train_samples)]
                try:
                    row = train_step(model, sample_dir, step, group, args)
                    train_log_rows.append(row)
                    append_csv_row(train_log_path, row, TRAIN_LOG_FIELDS)
                    if args.progress_every > 0 and (step == 1 or step % args.progress_every == 0 or step == args.max_steps):
                        write_partial_status(
                            output_dir,
                            {
                                "current_group": group.name,
                                "current_step": step,
                                "max_steps": args.max_steps,
                                "latest_total_loss": row["total_loss"],
                                "latest_l_center_x": row["l_center_x"],
                                "latest_carrier_regularization_total": row["carrier_regularization_total"],
                                "completed_groups": [g.name for g in selected_groups if (checkpoint_dir / f"{g.name}.pth").exists()],
                            },
                        )
                        print(
                            f"[Stage6A] {group.name} step {step}/{args.max_steps}: "
                            f"total_loss={row['total_loss']:.6f}, "
                            f"l_center_x={row['l_center_x']:.6f}, "
                            f"carrier_total={row['carrier_regularization_total']:.6f}",
                            flush=True,
                        )
                except Exception as exc:
                    group_failed = True
                    failed_row = {
                        "group": group.name,
                        "step": step,
                        "image_id": sample_dir.name,
                        "status": "failed",
                        "error_message": str(exc),
                    }
                    train_log_rows.append(failed_row)
                    append_csv_row(train_log_path, failed_row, TRAIN_LOG_FIELDS)
                    failed_cases.append(
                        {
                            "group": group.name,
                            "image_id": sample_dir.name,
                            "failure_stage": "training",
                            "error_message": str(exc),
                        }
                    )
                    break

            if not group_failed:
                save_checkpoint(model, checkpoint_dir / f"{group.name}.pth")
                print(f"[Stage6A] Evaluating group {group.name} on {len(val_samples)} samples", flush=True)
                group_eval_rows, group_vis_paths = evaluate_group(model, group, val_samples, output_dir, args)
                eval_rows.extend(group_eval_rows)
                for eval_row in group_eval_rows:
                    append_csv_row(eval_metrics_path, eval_row, EVAL_FIELDS)
                all_vis_paths.extend(group_vis_paths)
                for row in group_eval_rows:
                    if row["status"] != "ok":
                        failed_cases.append(
                            {
                                "group": group.name,
                                "image_id": str(row["image_id"]),
                                "failure_stage": "evaluation",
                                "error_message": str(row["error_message"]),
                            }
                        )
                collapse_reason = collapse_detected(group_eval_rows, group.name)
                if collapse_reason is not None:
                    failed_cases.append(
                        {
                            "group": group.name,
                            "image_id": "",
                            "failure_stage": "collapse_check",
                            "error_message": collapse_reason,
                        }
                    )
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    write_csv(output_dir / "stage6a_training_log.csv", train_log_rows, TRAIN_LOG_FIELDS)
    write_csv(output_dir / "stage6a_eval_metrics.csv", eval_rows, EVAL_FIELDS)
    summary = summarize_eval(eval_rows, failed_cases, selected_groups)
    summary["max_train_samples"] = args.max_train_samples
    summary["max_val_samples"] = args.max_val_samples
    summary["max_steps"] = args.max_steps
    summary["batch_size"] = args.batch_size
    summary["learning_rate"] = args.lr
    summary["groups_run"] = [group.name for group in selected_groups]
    summary["contact_sheet"] = build_contact_sheet(all_vis_paths, output_dir / "stage6a_contact_sheet.png", args.contact_sheet_samples)
    (output_dir / "stage6a_group_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "stage6a_failed_cases.json").write_text(json.dumps(failed_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved training log: {output_dir / 'stage6a_training_log.csv'}")
    print(f"Saved eval metrics: {output_dir / 'stage6a_eval_metrics.csv'}")
    print(f"Saved group summary: {output_dir / 'stage6a_group_summary.json'}")
    print(f"Saved failed cases: {output_dir / 'stage6a_failed_cases.json'}")
    print(f"Saved contact sheet: {output_dir / 'stage6a_contact_sheet.png'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
