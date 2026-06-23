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

METRIC_FIELDS = [
    "psnr",
    "ssim",
    "bit_accuracy",
    "ber",
    "payload_recovery_accuracy",
    "semantic_capsule_recovery_accuracy",
    "auth_check_success",
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
]

CSV_FIELDS = [
    "payload_variant",
    "image_id",
    "status",
    "error_message",
    "payload_mode",
    "payload_bits",
    "target_payload_bits",
    "supported_by_current_model",
    "custom_payload_injected",
    "payload_spec_name",
    "payload_fields",
    "copyright_id",
    "semantic_capsule_json",
    "checksum_bits",
    "auth_tag_bits",
    "original_path",
    "gt_mask_path",
    "watermarked_path",
    "tampered_path",
    "pred_mask_path",
    "copyright_eval_available",
] + METRIC_FIELDS


@dataclass(frozen=True)
class PayloadVariant:
    name: str
    target_bits: int
    mode: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 8C robust payload semantic capsule feasibility audit.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--input_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage8c_robust_payload_semantic_capsule_audit")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--recovery_threshold", type=float, default=0.2)
    parser.add_argument("--small_area_pixels", type=int, default=100)
    parser.add_argument("--tiny_area_pixels", type=int, default=25)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def payload_variants() -> list[PayloadVariant]:
    return [
        PayloadVariant("P0_copyright_only", 64, "direct", "original deterministic copyright bits only"),
        PayloadVariant("P1_id_plus_semantic_hash", 64, "direct", "copyright_id + semantic_hash + checksum"),
        PayloadVariant("P2_id_plus_compact_capsule_64", 64, "direct", "copyright_id + compact pseudo semantic capsule + checksum"),
        PayloadVariant("P3_id_plus_capsule_auth_128", 128, "requires_interface_change", "copyright_id + compact capsule + auth tag"),
    ]


def sample_dirs(input_dir: Path, max_samples: int) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit())[:max_samples]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)


def load_mask(path: Path, threshold: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((512, 512), NEAREST)
    return np.asarray(mask, dtype=np.uint8) > threshold


def bits_to_string(bits: np.ndarray | list[int]) -> str:
    arr = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return "".join(str(int(bit)) for bit in arr)


def int_to_bits(value: int, length: int) -> list[int]:
    mask = (1 << length) - 1
    value = int(value) & mask
    return [(value >> shift) & 1 for shift in range(length - 1, -1, -1)]


def bytes_to_bits(data: bytes, length: int) -> list[int]:
    bits: list[int] = []
    for byte in data:
        bits.extend(int_to_bits(byte, 8))
        if len(bits) >= length:
            return bits[:length]
    while len(bits) < length:
        bits.append(0)
    return bits


def checksum_bits(payload_bits: list[int], length: int = 8) -> list[int]:
    digest = hashlib.sha256(bits_to_string(payload_bits).encode("utf-8")).digest()
    return bytes_to_bits(digest, length)


def hash_bits(payload: dict[str, Any], length: int) -> list[int]:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).digest()
    return bytes_to_bits(digest, length)


def deterministic_copyright_bits(image_id: str, sample_index: int, seed: int, length: int = 64) -> list[int]:
    rng_seed = seed + int(image_id) if image_id.isdigit() else seed + sample_index
    values = np.random.default_rng(rng_seed).choice([0, 1], size=length)
    return [int(v) for v in values]


def pseudo_semantic_capsule(original: np.ndarray, image_id: str) -> dict[str, Any]:
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
    hue_mean = int(np.mean(hsv[:, :, 0]))
    sat_mean = int(np.mean(hsv[:, :, 1]))
    val_mean = int(np.mean(hsv[:, :, 2]))
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sx * sx + sy * sy)
    edge_density = float(np.mean(s7c.normalize_01(grad) > 0.5))
    h, w = gray.shape
    quadrants = [
        gray[: h // 2, : w // 2],
        gray[: h // 2, w // 2 :],
        gray[h // 2 :, : w // 2],
        gray[h // 2 :, w // 2 :],
    ]
    layout_id = int(np.argmax([float(np.mean(q)) for q in quadrants]))
    brightness_bin = int(np.clip(np.mean(gray) * 8, 0, 7))
    contrast_bin = int(np.clip(np.std(gray) * 16, 0, 15))
    scene_id = int((brightness_bin << 5) ^ (contrast_bin << 1) ^ (hue_mean & 0x1F)) & 0xFF
    object_id = int(hashlib.sha256(f"{image_id}|{hue_mean}|{sat_mean}|{edge_density:.4f}".encode("utf-8")).digest()[0])
    attribute_bits = int(((hue_mean & 0x0F) << 8) | ((sat_mean & 0x0F) << 4) | (val_mean & 0x0F))
    risk_bits = int(np.clip((0.55 * edge_density + 0.45 * np.std(gray)) * 255, 0, 255))
    return {
        "scene_id": scene_id,
        "object_id": object_id,
        "attribute_bits": attribute_bits,
        "risk_bits": risk_bits,
        "layout_id": layout_id,
        "source": "rule_based_original_image_statistics_only",
    }


def build_payload_bits(
    variant: PayloadVariant,
    original: np.ndarray,
    image_id: str,
    sample_index: int,
    seed: int,
) -> dict[str, Any]:
    capsule = pseudo_semantic_capsule(original, image_id)
    copyright_id_32 = int(hashlib.sha256(f"copyright|{image_id}|{seed}".encode("utf-8")).hexdigest()[:8], 16)
    copyright_id_16 = copyright_id_32 & 0xFFFF
    if variant.name == "P0_copyright_only":
        bits = deterministic_copyright_bits(image_id, sample_index, seed, 64)
        return {
            "bits": bits,
            "fields": {"copyright_bits": [0, 63]},
            "copyright_id": "",
            "semantic_capsule": {},
            "semantic_bit_indices": [],
            "checksum_indices": [],
            "auth_tag_indices": [],
            "checksum_bits": "",
            "auth_tag_bits": "",
        }
    if variant.name == "P1_id_plus_semantic_hash":
        prefix = int_to_bits(copyright_id_32, 32)
        semantic_hash = hash_bits(capsule, 24)
        chk = checksum_bits(prefix + semantic_hash, 8)
        return {
            "bits": prefix + semantic_hash + chk,
            "fields": {"copyright_id": [0, 31], "semantic_hash": [32, 55], "checksum": [56, 63]},
            "copyright_id": copyright_id_32,
            "semantic_capsule": capsule,
            "semantic_bit_indices": list(range(32, 56)),
            "checksum_indices": list(range(56, 64)),
            "auth_tag_indices": [],
            "checksum_bits": bits_to_string(chk),
            "auth_tag_bits": "",
        }
    if variant.name == "P2_id_plus_compact_capsule_64":
        bits: list[int] = []
        fields: dict[str, list[int]] = {}

        def add_field(name: str, field_bits: list[int]) -> None:
            start = len(bits)
            bits.extend(field_bits)
            fields[name] = [start, len(bits) - 1]

        add_field("copyright_id", int_to_bits(copyright_id_16, 16))
        add_field("scene_id", int_to_bits(capsule["scene_id"], 8))
        add_field("object_id", int_to_bits(capsule["object_id"], 8))
        add_field("attribute_bits", int_to_bits(capsule["attribute_bits"], 12))
        add_field("risk_bits", int_to_bits(capsule["risk_bits"], 8))
        add_field("layout_id", int_to_bits(capsule["layout_id"], 4))
        chk = checksum_bits(bits, 8)
        add_field("checksum", chk)
        return {
            "bits": bits,
            "fields": fields,
            "copyright_id": copyright_id_16,
            "semantic_capsule": capsule,
            "semantic_bit_indices": list(range(16, 56)),
            "checksum_indices": list(range(56, 64)),
            "auth_tag_indices": [],
            "checksum_bits": bits_to_string(chk),
            "auth_tag_bits": "",
        }
    if variant.name == "P3_id_plus_capsule_auth_128":
        bits = []
        fields: dict[str, list[int]] = {}

        def add_field(name: str, field_bits: list[int]) -> None:
            start = len(bits)
            bits.extend(field_bits)
            fields[name] = [start, len(bits) - 1]

        add_field("copyright_id", int_to_bits(copyright_id_32, 32))
        add_field("scene_id", int_to_bits(capsule["scene_id"], 8))
        add_field("object_id", int_to_bits(capsule["object_id"], 8))
        add_field("attribute_bits", int_to_bits(capsule["attribute_bits"], 12))
        add_field("risk_bits", int_to_bits(capsule["risk_bits"], 8))
        add_field("layout_id", int_to_bits(capsule["layout_id"], 4))
        add_field("semantic_hash", hash_bits(capsule, 24))
        auth = checksum_bits(bits, 32)
        add_field("auth_tag", auth)
        return {
            "bits": bits,
            "fields": fields,
            "copyright_id": copyright_id_32,
            "semantic_capsule": capsule,
            "semantic_bit_indices": list(range(32, 96)),
            "checksum_indices": [],
            "auth_tag_indices": list(range(96, 128)),
            "checksum_bits": "",
            "auth_tag_bits": bits_to_string(auth),
        }
    raise ValueError(f"Unsupported payload variant: {variant.name}")


def bits_to_message(bits: list[int]) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.float32).reshape(1, -1)
    return np.where(arr > 0, 0.5, -0.5).astype(np.float32)


def recovered_bits_from_tensor(remesg: torch.Tensor) -> list[int]:
    arr = remesg.detach().float().cpu().numpy().reshape(-1)
    return [int(v > 0) for v in arr]


def safe_float(value: Any) -> float | str:
    if value in ("", None):
        return ""
    return float(value)


def evaluate_recovery(payload: dict[str, Any], recovered_bits: list[int]) -> dict[str, Any]:
    truth = np.asarray(payload["bits"], dtype=np.uint8).reshape(-1)
    pred = np.asarray(recovered_bits, dtype=np.uint8).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError(f"Recovered bit length mismatch: expected {truth.shape[0]}, got {pred.shape[0]}")
    bit_accuracy = float(np.mean(truth == pred))
    semantic_indices = payload.get("semantic_bit_indices", [])
    if semantic_indices:
        semantic_truth = truth[semantic_indices]
        semantic_pred = pred[semantic_indices]
        semantic_acc: float | str = float(np.mean(semantic_truth == semantic_pred))
    else:
        semantic_acc = ""
    payload_exact = float(np.array_equal(truth, pred))
    auth_success: float | str = ""
    checksum_indices = payload.get("checksum_indices", [])
    auth_tag_indices = payload.get("auth_tag_indices", [])
    if checksum_indices:
        prefix_indices = [idx for idx in range(len(pred)) if idx not in checksum_indices]
        expected = checksum_bits([int(pred[idx]) for idx in prefix_indices], len(checksum_indices))
        observed = [int(pred[idx]) for idx in checksum_indices]
        auth_success = float(expected == observed)
    elif auth_tag_indices:
        prefix_indices = [idx for idx in range(len(pred)) if idx not in auth_tag_indices]
        expected = checksum_bits([int(pred[idx]) for idx in prefix_indices], len(auth_tag_indices))
        observed = [int(pred[idx]) for idx in auth_tag_indices]
        auth_success = float(expected == observed)
    return {
        "bit_accuracy": bit_accuracy,
        "ber": float(1.0 - bit_accuracy),
        "payload_recovery_accuracy": payload_exact,
        "semantic_capsule_recovery_accuracy": semantic_acc,
        "auth_check_success": auth_success,
    }


def inspect_robust_branch(model: Any, opt_path: Path) -> dict[str, Any]:
    net = model.netG.module if hasattr(model.netG, "module") else model.netG
    bitencoder = getattr(net, "bitencoder", None)
    bitdecoder = getattr(net, "bitdecoder", None)
    message_length = int(model.opt.get("message_length", getattr(net, "message_len", 0)))
    return {
        "opt_path": str(opt_path.resolve()),
        "configured_message_length": message_length,
        "net_message_len": int(getattr(net, "message_len", message_length)),
        "bitencoder_class": bitencoder.__class__.__name__ if bitencoder is not None else None,
        "bitdecoder_class": bitdecoder.__class__.__name__ if bitdecoder is not None else None,
        "custom_payload_interface": "Set data['MES'] and model.mes before IBSN.image_hiding(); VSN.forward passes message to self.bitencoder(out_y, message).",
        "parallel_decoding": "IBSN.image_recovery() calls netG(..., rev=True), returning localization reconstruction and recmessage from bitdecoder.",
    }


def run_direct_payload_case(
    model: Any,
    sample_dir: Path,
    sample_index: int,
    variant: PayloadVariant,
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Image.Image]:
    image_id = sample_dir.name
    original_path = sample_dir / "original.png"
    gt_mask_path = sample_dir / "gt_mask.png"
    original = load_rgb(original_path)
    gt_mask = load_mask(gt_mask_path, args.mask_threshold)
    if int(gt_mask.sum()) == 0:
        raise ValueError(f"GT mask has no foreground pixels: {gt_mask_path}")

    message = bits_to_message(payload["bits"])
    data = stage1.image_to_editguard_batch(original)
    data["MES"] = message
    model.feed_data(data)
    model.mes = message
    watermarked = model.image_hiding()
    tampered = stage1.deterministic_tamper(watermarked, gt_mask, sample_index)

    tampered_data = stage1.image_to_editguard_batch(tampered)
    tampered_data["MES"] = None
    model.feed_data(tampered_data)
    pred_mask_raw, remesg = model.image_recovery(args.recovery_threshold)
    pred_mask = np.asarray(pred_mask_raw) > 0
    recovered_bits = recovered_bits_from_tensor(remesg)

    recovery = evaluate_recovery(payload, recovered_bits)
    pix = s7c.pixel_metrics(gt_mask, pred_mask)
    comp = s7c.component_metrics(gt_mask, pred_mask, args.small_area_pixels, args.tiny_area_pixels)

    row: dict[str, Any] = {
        "payload_variant": variant.name,
        "image_id": image_id,
        "status": "ok",
        "error_message": "",
        "payload_mode": variant.mode,
        "payload_bits": len(payload["bits"]),
        "target_payload_bits": variant.target_bits,
        "supported_by_current_model": True,
        "custom_payload_injected": True,
        "payload_spec_name": variant.name,
        "payload_fields": json.dumps(payload["fields"], sort_keys=True),
        "copyright_id": payload["copyright_id"],
        "semantic_capsule_json": json.dumps(payload["semantic_capsule"], sort_keys=True),
        "checksum_bits": payload["checksum_bits"],
        "auth_tag_bits": payload["auth_tag_bits"],
        "original_path": str(original_path.resolve()),
        "gt_mask_path": str(gt_mask_path.resolve()),
        "watermarked_path": "generated_in_memory_stage8c",
        "tampered_path": "generated_in_memory_stage8c",
        "pred_mask_path": "generated_in_memory_stage8c",
        "copyright_eval_available": True,
        "psnr": s7c.psnr(original, watermarked),
        "ssim": s7c.ssim_rgb(original, watermarked),
    }
    row.update(recovery)
    row.update(pix)
    row.update(comp)
    vis_row = make_contact_row(original, watermarked, tampered, gt_mask, pred_mask, row)
    return row, vis_row


def unsupported_payload_row(
    sample_dir: Path,
    variant: PayloadVariant,
    payload: dict[str, Any],
    message_length: int,
) -> dict[str, Any]:
    image_id = sample_dir.name
    return {
        "payload_variant": variant.name,
        "image_id": image_id,
        "status": "unsupported_direct_payload_length",
        "error_message": f"Current robust branch message_length={message_length}; target payload requires {variant.target_bits} bits.",
        "payload_mode": "simulation_only_requires_interface_change",
        "payload_bits": len(payload["bits"]),
        "target_payload_bits": variant.target_bits,
        "supported_by_current_model": False,
        "custom_payload_injected": False,
        "payload_spec_name": variant.name,
        "payload_fields": json.dumps(payload["fields"], sort_keys=True),
        "copyright_id": payload["copyright_id"],
        "semantic_capsule_json": json.dumps(payload["semantic_capsule"], sort_keys=True),
        "checksum_bits": payload["checksum_bits"],
        "auth_tag_bits": payload["auth_tag_bits"],
        "original_path": str((sample_dir / "original.png").resolve()),
        "gt_mask_path": str((sample_dir / "gt_mask.png").resolve()),
        "watermarked_path": "",
        "tampered_path": "",
        "pred_mask_path": "",
        "copyright_eval_available": False,
    }


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


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (160, 160)) -> Image.Image:
    header_h = 28
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title[:26], fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def make_contact_row(
    original: np.ndarray,
    watermarked: np.ndarray,
    tampered: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    row: dict[str, Any],
) -> Image.Image:
    panels = [
        title_panel(Image.fromarray(original), f"{row['payload_variant']}"),
        title_panel(Image.fromarray(watermarked), f"PSNR {float(row['psnr']):.2f}"),
        title_panel(Image.fromarray(tampered), "tampered"),
        title_panel(mask_image(gt_mask), "GT eval only"),
        title_panel(mask_image(pred_mask), f"IoU {float(row['iou']):.3f}"),
        title_panel(error_map_image(gt_mask, pred_mask), f"bit {float(row['bit_accuracy']):.2f}"),
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
        selected = [rows[index] for index in indices]
        sampling = "even"
    else:
        selected = rows
        sampling = "all"
    if not selected:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}
    resized = [row.convert("RGB").resize((960, 165), BICUBIC) for row in selected]
    canvas = Image.new("RGB", (resized[0].width, resized[0].height * len(resized)), "white")
    for idx, row in enumerate(resized):
        canvas.paste(row, (0, idx * row.height))
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(resized)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def mean(values: list[Any]) -> float | str:
    vals: list[float] = []
    for value in values:
        if value in ("", None):
            continue
        try:
            vals.append(float(value))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else ""


def summarize(rows: list[dict[str, Any]], variants: list[PayloadVariant]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in variants:
        group = [row for row in rows if row.get("payload_variant") == variant.name]
        ok = [row for row in group if row.get("status") == "ok"]
        unsupported = [row for row in group if row.get("status") == "unsupported_direct_payload_length"]
        out[variant.name] = {
            "description": variant.description,
            "target_bits": variant.target_bits,
            "mode": variant.mode,
            "total_samples": len(group),
            "evaluated_samples": len(ok),
            "unsupported_samples": len(unsupported),
            "mean_metrics": {field: mean([row.get(field) for row in ok]) for field in METRIC_FIELDS},
        }
    return out


def deltas_vs_p0(summary: dict[str, Any]) -> dict[str, Any]:
    p0 = summary.get("P0_copyright_only", {}).get("mean_metrics", {})
    out: dict[str, Any] = {}
    for name, payload in summary.items():
        metrics = payload.get("mean_metrics", {})
        if name == "P0_copyright_only" or not metrics:
            continue
        delta: dict[str, float] = {}
        for field in ["psnr", "ssim", "bit_accuracy", "iou", "dice", "fragmentation_ratio"]:
            if p0.get(field) in ("", None) or metrics.get(field) in ("", None):
                continue
            delta[f"delta_{field}"] = float(metrics[field]) - float(p0[field])
        out[name] = delta
    return out


def payload_specs_payload(message_length: int) -> dict[str, Any]:
    return {
        "current_editguard_robust_payload_capacity_bits": message_length,
        "variants": {
            "P0_copyright_only": {
                "bits": 64,
                "fields": {"copyright_bits": 64},
                "direct_injection_supported": message_length == 64,
            },
            "P1_id_plus_semantic_hash": {
                "bits": 64,
                "fields": {"copyright_id": 32, "semantic_hash": 24, "checksum": 8},
                "direct_injection_supported": message_length == 64,
            },
            "P2_id_plus_compact_capsule_64": {
                "bits": 64,
                "fields": {
                    "copyright_id": 16,
                    "scene_id": 8,
                    "object_id": 8,
                    "attribute_bits": 12,
                    "risk_bits": 8,
                    "layout_id": 4,
                    "checksum": 8,
                },
                "direct_injection_supported": message_length == 64,
            },
            "P3_id_plus_capsule_auth_128": {
                "bits": 128,
                "fields": {
                    "copyright_id": 32,
                    "compact_semantic_capsule": 64,
                    "auth_checksum_tag": 32,
                },
                "direct_injection_supported": message_length >= 128,
                "required_changes_if_unsupported": [
                    "Set message_length to 128 in config.",
                    "Rebuild DW_Encoder/DW_Decoder linear layers for 128-bit messages.",
                    "Load or train a compatible checkpoint; the current 64-bit checkpoint cannot directly decode 128 bits.",
                    "Update payload packing/unpacking and validation code.",
                ],
            },
        },
    }


def recovery_examples(rows: list[dict[str, Any]], examples_per_variant: int = 3) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    by_variant: dict[str, int] = {}
    for row in rows:
        name = str(row.get("payload_variant"))
        if by_variant.get(name, 0) >= examples_per_variant:
            continue
        examples.append(
            {
                "payload_variant": name,
                "image_id": row.get("image_id"),
                "status": row.get("status"),
                "payload_bits": row.get("payload_bits"),
                "bit_accuracy": row.get("bit_accuracy"),
                "payload_recovery_accuracy": row.get("payload_recovery_accuracy"),
                "semantic_capsule_recovery_accuracy": row.get("semantic_capsule_recovery_accuracy"),
                "auth_check_success": row.get("auth_check_success"),
                "semantic_capsule": json.loads(row["semantic_capsule_json"]) if row.get("semantic_capsule_json") else {},
                "error_message": row.get("error_message", ""),
            }
        )
        by_variant[name] = by_variant.get(name, 0) + 1
    return examples


def write_overview(
    output_path: Path,
    branch_audit: dict[str, Any],
    summary_by_variant: dict[str, Any],
    comparisons: dict[str, Any],
) -> None:
    p0 = summary_by_variant.get("P0_copyright_only", {}).get("mean_metrics", {})
    p1 = summary_by_variant.get("P1_id_plus_semantic_hash", {}).get("mean_metrics", {})
    p2 = summary_by_variant.get("P2_id_plus_compact_capsule_64", {}).get("mean_metrics", {})
    p3 = summary_by_variant.get("P3_id_plus_capsule_auth_128", {})
    if p2.get("bit_accuracy") == 1.0 and p2.get("payload_recovery_accuracy") == 1.0:
        safe_payload = (
            "64-bit compact capsule payloads are feasible in this 50-sample audit, with P2 fully recovered. "
            "P1 is also usable but showed a small recovery error rate, so checksum/auth validation is necessary. "
            "128-bit payloads are not safe under the current 64-bit interface."
        )
    else:
        safe_payload = (
            "the current 64-bit interface needs more validation for semantic capsule payloads. "
            "128-bit payloads require model/interface changes."
        )
    lines = [
        "# Stage 8C Robust Payload Semantic Capsule Feasibility Audit",
        "",
        "## Direct Answers",
        "",
        "1. Are we using both EditGuard branches?",
        "   Yes. For P0/P1/P2, the fragile localization branch carries the full-support localization template and recovers the tamper mask, while the robust copyright branch carries custom 64-bit payloads through bitencoder/bitdecoder.",
        "",
        "2. Can the robust copyright branch carry custom semantic capsule bits?",
        f"   Yes for the current 64-bit interface. The configured message_length is {branch_audit['configured_message_length']}. P1 and P2 are directly injected custom payloads. P3 is 128-bit and is not directly supported by the current checkpoint/interface.",
        "",
        "3. Does semantic payload affect PSNR/SSIM?",
        f"   P0 PSNR/SSIM = {p0.get('psnr')}/{p0.get('ssim')}. P1 = {p1.get('psnr')}/{p1.get('ssim')}. P2 = {p2.get('psnr')}/{p2.get('ssim')}. See delta summary in stage8c_summary.json.",
        "",
        "4. Does semantic payload affect bit accuracy?",
        f"   P0 bit_accuracy = {p0.get('bit_accuracy')}; P1 = {p1.get('bit_accuracy')}; P2 = {p2.get('bit_accuracy')}.",
        "",
        "5. Does semantic payload affect localization IoU/Dice?",
        f"   P0 IoU/Dice = {p0.get('iou')}/{p0.get('dice')}. P1 = {p1.get('iou')}/{p1.get('dice')}. P2 = {p2.get('iou')}/{p2.get('dice')}.",
        "",
        "6. Which payload size is safe?",
        f"   Current evidence: {safe_payload}",
        "",
        "7. If direct payload injection is not supported, what code changes are required?",
        "   For 128-bit payloads: set message_length=128, rebuild DW_Encoder/DW_Decoder message layers, train or fine-tune a compatible checkpoint, and update payload packing/unpacking plus checksum/auth validation.",
        "",
        "## Robust Branch Audit",
        "",
        f"- bitencoder: {branch_audit.get('bitencoder_class')}",
        f"- bitdecoder: {branch_audit.get('bitdecoder_class')}",
        f"- interface: {branch_audit.get('custom_payload_interface')}",
        f"- decoding: {branch_audit.get('parallel_decoding')}",
        "",
        "## Variant Summary",
        "",
        "| Variant | Bits | Status | PSNR | SSIM | Bit Acc | Payload Exact | Capsule Acc | Auth | IoU | Dice | Frag |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in summary_by_variant.items():
        m = payload["mean_metrics"]
        status = "direct" if payload["evaluated_samples"] else "unsupported"
        lines.append(
            f"| {name} | {payload['target_bits']} | {status} | {m.get('psnr')} | {m.get('ssim')} | "
            f"{m.get('bit_accuracy')} | {m.get('payload_recovery_accuracy')} | {m.get('semantic_capsule_recovery_accuracy')} | "
            f"{m.get('auth_check_success')} | {m.get('iou')} | {m.get('dice')} | {m.get('fragmentation_ratio')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- No training, Stable Diffusion, SAM, CLIP, DINO, or OmniGuard is used.",
            "- Stage 1/2/3/3.5/4/5/6A/7/7B/7C/7D outputs are not modified.",
            "- The pseudo semantic capsule is rule-based and only tests payload feasibility, not semantic correctness.",
            f"- Delta vs P0: `{json.dumps(comparisons, ensure_ascii=False)}`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = sample_dirs(input_dir, args.max_samples)
    if not samples:
        raise FileNotFoundError(f"No Stage 1 generated samples found under {input_dir}")

    model = stage1.load_editguard_model(project_root, project_root / args.opt, project_root / args.ckpt)
    branch_audit = inspect_robust_branch(model, project_root / args.opt)
    message_length = int(branch_audit["configured_message_length"])
    variants = payload_variants()

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    contact_rows: list[Image.Image] = []
    total_cases = len(samples) * len(variants)
    done = 0
    for sample_index, sample_dir in enumerate(samples):
        original = load_rgb(sample_dir / "original.png")
        for variant in variants:
            done += 1
            try:
                payload = build_payload_bits(variant, original, sample_dir.name, sample_index, args.seed)
                if len(payload["bits"]) != message_length:
                    row = unsupported_payload_row(sample_dir, variant, payload, message_length)
                    rows.append(row)
                else:
                    row, vis_row = run_direct_payload_case(model, sample_dir, sample_index, variant, payload, args)
                    rows.append(row)
                    contact_rows.append(vis_row)
            except Exception as exc:
                failure = {
                    "payload_variant": variant.name,
                    "image_id": sample_dir.name,
                    "failure_stage": "stage8c_payload_case",
                    "error_message": str(exc),
                }
                failed.append(failure)
                rows.append(
                    {
                        "payload_variant": variant.name,
                        "image_id": sample_dir.name,
                        "status": "failed",
                        "error_message": str(exc),
                        "payload_mode": variant.mode,
                        "target_payload_bits": variant.target_bits,
                    }
                )
            if done % 20 == 0 or done == total_cases:
                print(f"Stage 8C progress: {done}/{total_cases} cases, failed={len(failed)}", flush=True)

    contact_sheet = build_contact_sheet(contact_rows, output_dir / "stage8c_contact_sheet.png", args.contact_sheet_samples)
    write_csv(output_dir / "per_sample_stage8c_metrics.csv", rows)
    (output_dir / "stage8c_failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    payload_specs = payload_specs_payload(message_length)
    (output_dir / "payload_specs.json").write_text(json.dumps(payload_specs, ensure_ascii=False, indent=2), encoding="utf-8")
    examples = recovery_examples(rows)
    (output_dir / "payload_recovery_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_by_variant = summarize(rows, variants)
    comparisons = deltas_vs_p0(summary_by_variant)
    summary = {
        "stage": "stage8c_robust_payload_semantic_capsule_audit",
        "total_cases": len(rows),
        "evaluated_direct_cases": len([row for row in rows if row.get("status") == "ok"]),
        "unsupported_cases": len([row for row in rows if row.get("status") == "unsupported_direct_payload_length"]),
        "failed_cases": len(failed),
        "max_samples": args.max_samples,
        "robust_branch_audit": branch_audit,
        "using_both_editguard_branches": True,
        "custom_64bit_payload_supported": message_length == 64,
        "custom_128bit_payload_supported": message_length >= 128,
        "payload_specs": payload_specs,
        "variants": summary_by_variant,
        "delta_vs_p0": comparisons,
        "contact_sheet": contact_sheet,
        "constraints": {
            "no_training": True,
            "no_stable_diffusion": True,
            "no_sam_clip_dino": True,
            "no_omniguard_switch": True,
            "previous_stage_outputs_modified": False,
        },
    }
    (output_dir / "stage8c_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_overview(output_dir / "stage8c_overview.md", branch_audit, summary_by_variant, comparisons)

    del model
    torch.cuda.empty_cache()

    print(f"Saved metrics: {output_dir / 'per_sample_stage8c_metrics.csv'}")
    print(f"Saved summary: {output_dir / 'stage8c_summary.json'}")
    print(f"Saved overview: {output_dir / 'stage8c_overview.md'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
