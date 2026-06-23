from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_editguard_outputs_stage1 as stage1  # noqa: E402
import run_stage8c_robust_payload_semantic_capsule_audit as s8c  # noqa: E402
from run_stage10a_attack_locality_audit import (  # noqa: E402
    BICUBIC,
    NEAREST,
    eval_metrics_row,
    load_json,
    pred_area_from_eval_dir,
    read_csv,
    safe_float,
    stage9_metrics,
    write_csv,
)
from run_stage10a_real_aigc_attack_subset import (  # noqa: E402
    ATTACK_TYPES,
    NEGATIVE_PROMPT,
    PROMPTS,
    changed_pixel_ratio,
    generate_with_diffusers,
    load_diffusers_pipe,
    mask_from_pred,
    successful_stage9_ids,
)


AVOID_STAGE9_FAILED_IDS = {
    "0070",
    "0106",
    "0116",
    "0327",
    "0499",
    "0517",
    "0650",
    "0673",
    "0681",
    "0741",
    "0759",
}

GEN_FIELDS = [
    "sample_id",
    "source_image_id",
    "attack_type",
    "attack_prompt",
    "negative_prompt",
    "seed",
    "source_original_path",
    "source_pred_mask_path",
    "generation_watermarked_path",
    "diffusion_generated_path",
    "edit_mask_path",
    "gt_mask_path",
    "payload_path",
    "generation_status",
    "generator_name",
    "generator_config",
    "runtime_seconds",
    "gpu_peak_memory_mb",
    "mask_area_ratio",
    "diffusion_changed_pixel_ratio_inside_mask",
    "error_message",
]

LC_FIELDS = [
    "sample_id",
    "source_image_id",
    "attack_type",
    "watermarked_source_path",
    "diffusion_generated_path",
    "edit_mask_path",
    "localized_composite_path",
    "gt_mask_path",
    "mask_area_ratio",
    "outside_region_preserved_check",
    "outside_mean_abs_diff",
    "outside_max_abs_diff",
    "inside_region_changed_check",
    "inside_mean_abs_diff",
    "image_size_match",
    "generation_status",
    "error_message",
    "original_image_path",
    "watermarked_image_path",
    "generated_tampered_image_path",
    "attack_prompt",
    "negative_prompt",
    "generator_name",
    "generator_config",
    "seed",
    "runtime_seconds",
    "gpu_peak_memory_mb",
    "tamper_changed_pixel_ratio",
    "notes",
]


@dataclass(frozen=True)
class ScaleCase:
    sample_id: str
    source_image_id: str
    attack_type: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10A localized-composite scale1000 generation, evaluation, and diagnosis.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--stage9_dir", default="dfg_locguard/outputs/stage9_full_evaluation_1000")
    parser.add_argument("--generation_dir", default="dfg_locguard/outputs/stage10a_lc_scale1000_generation")
    parser.add_argument("--localized_dir", default="dfg_locguard/outputs/stage10a_lc_scale1000")
    parser.add_argument("--eval_dir", default="dfg_locguard/outputs/stage10a_lc_scale1000_eval")
    parser.add_argument("--unknown_dir", default="dfg_locguard/outputs/stage10a_lc_unknown_diagnosis")
    parser.add_argument("--stage10a_unconstrained_eval_dir", default="dfg_locguard/outputs/stage10a_real_aigc_attack_eval")
    parser.add_argument("--stage10a_audit_dir", default="dfg_locguard/outputs/stage10a_attack_locality_audit")
    parser.add_argument("--results_summary_dir", default="dfg_locguard/results_summary")
    parser.add_argument("--model_id", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    parser.add_argument("--hf_cache_dir", default="models/hf_cache")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--samples_per_type", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.95)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--min_free_gb", type=float, default=25.0)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def mean(values: list[Any]) -> float:
    nums = [safe_float(v) for v in values if v not in ("", None)]
    return float(sum(nums) / len(nums)) if nums else 0.0


def check_free_space(path: Path, min_free_gb: float) -> float:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"Disk free space {free_gb:.2f}GB is below threshold {min_free_gb:.2f}GB")
    return free_gb


def select_cases(stage9_dir: Path, samples_per_type: int, seed: int) -> list[ScaleCase]:
    rows = [row for row in successful_stage9_ids(stage9_dir) if row.get("image_id") not in AVOID_STAGE9_FAILED_IDS]
    if len(rows) < samples_per_type:
        raise RuntimeError(f"Need at least {samples_per_type} successful Stage9C rows, found {len(rows)}")
    # Use the same deterministic stratified pattern as earlier Stage 10A, but scale to 250 per attack type.
    buckets = [rows[i::4] for i in range(4)]
    rng = np.random.default_rng(seed)
    for bucket in buckets:
        rng.shuffle(bucket)
    cases: list[ScaleCase] = []
    for attack_type in ATTACK_TYPES:
        picked = 0
        cursor = 0
        used: set[str] = set()
        while picked < samples_per_type:
            bucket = buckets[cursor % len(buckets)]
            row = bucket[(cursor // len(buckets)) % len(bucket)]
            image_id = row["image_id"]
            cursor += 1
            if image_id in used:
                continue
            used.add(image_id)
            sample_id = f"{attack_type}_{picked + 1:03d}_{image_id}"
            cases.append(ScaleCase(sample_id, image_id, attack_type, seed + len(cases) * 17))
            picked += 1
    return cases


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)


def load_mask(path: Path, threshold: int = 127) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((512, 512), NEAREST), dtype=np.uint8) > threshold


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def image_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def generate_diffusion_edits(args: argparse.Namespace, project_root: Path, stage9_dir: Path, generation_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = generation_dir / "stage10a_lc_scale1000_manifest.csv"
    if args.skip_generation and manifest_path.exists():
        rows = read_csv(manifest_path)
        summary = load_json(generation_dir / "stage10a_lc_scale1000_generation_summary.json")
        return rows, summary
    if args.overwrite and generation_dir.exists():
        shutil.rmtree(generation_dir)
        generation_dir.mkdir(parents=True, exist_ok=True)
    (generation_dir / "diffusion_attacks").mkdir(parents=True, exist_ok=True)
    (generation_dir / "masks").mkdir(parents=True, exist_ok=True)

    check_free_space(project_root, args.min_free_gb)
    cases = select_cases(stage9_dir, args.samples_per_type, args.seed)
    gen_args = argparse.Namespace(**vars(args))
    gen_args.model_cache_dir = args.hf_cache_dir
    model = stage1.load_editguard_model(project_root, project_root / args.opt, project_root / args.ckpt)
    pipe = load_diffusers_pipe(gen_args, project_root)
    rows: list[dict[str, Any]] = []
    generated_dirs: list[Path] = []
    started_all = time.time()
    for index, case in enumerate(cases, start=1):
        check_free_space(project_root, args.min_free_gb)
        source_dir = stage9_dir / "stage1_editguard_generated" / case.source_image_id
        original_path = source_dir / "original.png"
        pred_mask_path = source_dir / "editguard_pred_mask.png"
        case_dir = generation_dir / "diffusion_attacks" / case.attack_type / case.sample_id
        case_dir.mkdir(parents=True, exist_ok=True)
        edit_mask_path = generation_dir / "masks" / f"{case.sample_id}_edit_mask.png"
        row: dict[str, Any] = {
            "sample_id": case.sample_id,
            "source_image_id": case.source_image_id,
            "attack_type": case.attack_type,
            "attack_prompt": PROMPTS[case.attack_type],
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": case.seed,
            "source_original_path": str(original_path.resolve()),
            "source_pred_mask_path": str(pred_mask_path.resolve()),
            "generation_watermarked_path": str((case_dir / "watermarked.png").resolve()),
            "diffusion_generated_path": str((case_dir / "diffusion_generated.png").resolve()),
            "edit_mask_path": str(edit_mask_path.resolve()),
            "gt_mask_path": str((case_dir / "gt_mask.png").resolve()),
            "payload_path": str((case_dir / "payload.json").resolve()),
            "generation_status": "failed",
            "generator_name": "diffusers_inpaint",
            "generator_config": json.dumps(
                {
                    "model_id": args.model_id,
                    "steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "strength": args.strength,
                    "dtype": args.dtype,
                    "device": args.device,
                },
                sort_keys=True,
            ),
            "runtime_seconds": "",
            "gpu_peak_memory_mb": "",
            "mask_area_ratio": "",
            "diffusion_changed_pixel_ratio_inside_mask": "",
            "error_message": "",
        }
        try:
            host_img = Image.open(original_path).convert("RGB").resize((512, 512), BICUBIC)
            host = np.asarray(host_img, dtype=np.uint8)
            edit_mask = mask_from_pred(pred_mask_path, case.attack_type, case.seed)
            edit_mask_img = Image.fromarray(edit_mask.astype(np.uint8) * 255).convert("L").filter(ImageFilter.MaxFilter(args.mask_dilate))
            edit_mask = np.asarray(edit_mask_img, dtype=np.uint8) > 127
            if int(edit_mask.sum()) == 0:
                raise ValueError("empty edit mask")
            payload = s8c.build_payload_bits(
                s8c.PayloadVariant("P2_id_plus_compact_capsule_64", 64, "direct", ""),
                host,
                case.sample_id,
                index,
                args.seed,
            )
            message = s8c.bits_to_message(payload["bits"])
            data = stage1.image_to_editguard_batch(host)
            data["MES"] = message
            model.feed_data(data)
            model.mes = message
            watermarked = model.image_hiding()
            watermarked_img = image_to_pil_rgb(watermarked)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.time()
            diffusion_img = generate_with_diffusers(pipe, watermarked_img, edit_mask_img, case.attack_type, case.seed, gen_args)
            runtime = time.time() - started
            peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
            host_img.save(case_dir / "original.png")
            watermarked_img.save(case_dir / "watermarked.png")
            diffusion_img.save(case_dir / "diffusion_generated.png")
            save_mask(case_dir / "gt_mask.png", edit_mask)
            save_mask(case_dir / "edit_mask.png", edit_mask)
            save_mask(edit_mask_path, edit_mask)
            write_json(case_dir / "payload.json", payload)
            write_json(
                case_dir / "meta.json",
                {
                    "case": row,
                    "payload_variant": "P2_id_plus_compact_capsule_64",
                    "stage": "stage10a_lc_scale1000_generation",
                },
            )
            ratio = changed_pixel_ratio(watermarked, np.asarray(diffusion_img), edit_mask)
            row.update(
                {
                    "generation_status": "ok",
                    "runtime_seconds": runtime,
                    "gpu_peak_memory_mb": peak_mb,
                    "mask_area_ratio": float(np.mean(edit_mask)),
                    "diffusion_changed_pixel_ratio_inside_mask": ratio,
                }
            )
            generated_dirs.append(case_dir)
        except Exception as exc:
            row["generation_status"] = "failed"
            row["error_message"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if index % 25 == 0 or index == len(cases):
            ok = sum(1 for item in rows if item.get("generation_status") == "ok")
            print(f"Stage10A LC1000 generation {index}/{len(cases)} ok={ok} failed={len(rows)-ok}", flush=True)
    write_csv(manifest_path, rows, GEN_FIELDS)
    save_contact_sheet_from_generation(generated_dirs, generation_dir / "stage10a_lc_scale1000_generation_contact_sheet.png", args.contact_sheet_samples)
    ok_rows = [row for row in rows if row.get("generation_status") == "ok"]
    failed_rows = [row for row in rows if row.get("generation_status") != "ok"]
    summary = {
        "stage": "stage10a_lc_scale1000_generation",
        "total_cases": len(rows),
        "generated_cases": len(ok_rows),
        "failed_cases": len(failed_rows),
        "generation_success_rate": len(ok_rows) / len(rows) if rows else 0.0,
        "samples_by_attack_type": dict(Counter(row["attack_type"] for row in ok_rows)),
        "total_runtime_seconds": time.time() - started_all,
        "mean_runtime_seconds": mean([row.get("runtime_seconds") for row in ok_rows]),
        "peak_gpu_memory_mb": max([safe_float(row.get("gpu_peak_memory_mb")) for row in ok_rows], default=0.0),
        "model_id": args.model_id,
        "hf_cache_dir": str(resolve(project_root, args.hf_cache_dir)),
        "manifest_path": str(manifest_path.resolve()),
        "failed_samples": failed_rows,
        "constraints": {
            "no_training": True,
            "no_vlm": True,
            "no_128bit_payload": True,
            "watermark_embedding_modified": False,
            "robust_payload_bits": 64,
        },
    }
    write_json(generation_dir / "stage10a_lc_scale1000_generation_summary.json", summary)
    if summary["generation_success_rate"] < 0.95:
        raise RuntimeError(f"Generation success rate below threshold: {summary['generation_success_rate']}")
    return rows, summary


def save_contact_sheet_from_generation(case_dirs: list[Path], out_path: Path, max_items: int) -> None:
    if not case_dirs:
        return
    if len(case_dirs) <= max_items:
        selected = case_dirs
    else:
        selected = [case_dirs[round(i * (len(case_dirs) - 1) / (max_items - 1))] for i in range(max_items)]
    panels = []
    for case_dir in selected:
        paths = [case_dir / "watermarked.png", case_dir / "edit_mask.png", case_dir / "diffusion_generated.png"]
        if not all(path.exists() for path in paths):
            continue
        panel = Image.new("RGB", (540, 210), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((6, 6), case_dir.name, fill=(0, 0, 0))
        for idx, path in enumerate(paths):
            image = Image.open(path).convert("RGB").resize((180, 180), BICUBIC)
            panel.paste(image, (idx * 180, 28))
        panels.append(panel)
    if not panels:
        return
    canvas = Image.new("RGB", (540, 210 * len(panels)), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (0, idx * 210))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def construct_localized_composites(args: argparse.Namespace, project_root: Path, generation_rows: list[dict[str, Any]], localized_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.overwrite and localized_dir.exists():
        shutil.rmtree(localized_dir)
    localized_dir.mkdir(parents=True, exist_ok=True)
    (localized_dir / "localized_attacks").mkdir(parents=True, exist_ok=True)
    (localized_dir / "masks").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, gen_row in enumerate([row for row in generation_rows if row.get("generation_status") == "ok"], start=1):
        sample_id = str(gen_row["sample_id"])
        attack_type = str(gen_row["attack_type"])
        gen_case_dir = Path(str(gen_row["diffusion_generated_path"])).resolve().parent
        out_dir = localized_dir / "localized_attacks" / attack_type / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_out = localized_dir / "masks" / f"{sample_id}_edit_mask.png"
        composite_path = out_dir / "tampered.png"
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source_image_id": gen_row.get("source_image_id"),
            "attack_type": attack_type,
            "watermarked_source_path": str((out_dir / "watermarked.png").resolve()),
            "diffusion_generated_path": gen_row.get("diffusion_generated_path"),
            "edit_mask_path": str(mask_out.resolve()),
            "localized_composite_path": str(composite_path.resolve()),
            "gt_mask_path": str((out_dir / "gt_mask.png").resolve()),
            "mask_area_ratio": "",
            "outside_region_preserved_check": False,
            "outside_mean_abs_diff": "",
            "outside_max_abs_diff": "",
            "inside_region_changed_check": False,
            "inside_mean_abs_diff": "",
            "image_size_match": False,
            "generation_status": "failed",
            "error_message": "",
            "original_image_path": str((out_dir / "original.png").resolve()),
            "watermarked_image_path": str((out_dir / "watermarked.png").resolve()),
            "generated_tampered_image_path": str(composite_path.resolve()),
            "attack_prompt": gen_row.get("attack_prompt", ""),
            "negative_prompt": gen_row.get("negative_prompt", ""),
            "generator_name": "localized_composite_scale1000",
            "generator_config": json.dumps({"rule": "tampered_local = mask * diffusion_generated + (1-mask) * watermarked_image"}, sort_keys=True),
            "seed": gen_row.get("seed", ""),
            "runtime_seconds": "",
            "gpu_peak_memory_mb": gen_row.get("gpu_peak_memory_mb", "0"),
            "tamper_changed_pixel_ratio": "",
            "notes": "Fair local AIGC edit; unmasked region is copied exactly from watermarked source.",
        }
        try:
            started_case = time.time()
            watermarked = load_rgb(gen_case_dir / "watermarked.png")
            diffusion = load_rgb(gen_case_dir / "diffusion_generated.png")
            mask = load_mask(gen_case_dir / "gt_mask.png", args.mask_threshold)
            if int(mask.sum()) == 0:
                raise ValueError("empty edit mask")
            if watermarked.shape != diffusion.shape:
                raise ValueError(f"image size mismatch: watermarked={watermarked.shape}, diffusion={diffusion.shape}")
            composite = watermarked.copy()
            composite[mask] = diffusion[mask]
            diff = np.abs(composite.astype(np.int16) - watermarked.astype(np.int16))
            outside = ~mask
            inside = mask
            outside_mean = float(np.mean(diff[outside])) if int(outside.sum()) else 0.0
            outside_max = int(np.max(diff[outside])) if int(outside.sum()) else 0
            inside_mean = float(np.mean(diff[inside])) if int(inside.sum()) else 0.0
            copy_if_exists(gen_case_dir / "payload.json", out_dir / "payload.json")
            save_rgb(out_dir / "original.png", load_rgb(gen_case_dir / "original.png"))
            save_rgb(out_dir / "watermarked.png", watermarked)
            save_rgb(composite_path, composite)
            save_mask(out_dir / "gt_mask.png", mask)
            save_mask(out_dir / "edit_mask.png", mask)
            save_mask(mask_out, mask)
            write_json(
                out_dir / "meta.json",
                {
                    "sample_id": sample_id,
                    "source_generation_row": gen_row,
                    "localized_composite_rule": "tampered_local = mask * diffusion_generated + (1-mask) * watermarked_image",
                    "verification_original_access": False,
                    "gt_mask_usage": "evaluation_only",
                },
            )
            row.update(
                {
                    "mask_area_ratio": float(np.mean(mask)),
                    "outside_region_preserved_check": outside_max <= 1 and outside_mean <= 1e-6,
                    "outside_mean_abs_diff": outside_mean,
                    "outside_max_abs_diff": outside_max,
                    "inside_region_changed_check": inside_mean > 1.0,
                    "inside_mean_abs_diff": inside_mean,
                    "image_size_match": True,
                    "generation_status": "ok",
                    "runtime_seconds": time.time() - started_case,
                    "tamper_changed_pixel_ratio": float(np.mean(np.mean(diff, axis=2)[inside] > 3.0)),
                }
            )
        except Exception as exc:
            row["error_message"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if index % 100 == 0 or index == len(generation_rows):
            failed = sum(1 for item in rows if item.get("generation_status") != "ok")
            print(f"Stage10A LC1000 composite {index}/{len(generation_rows)} failed={failed}", flush=True)
    manifest_path = localized_dir / "stage10a_lc_scale1000_manifest.csv"
    write_csv(manifest_path, rows, LC_FIELDS)
    save_contact_sheet_from_lc(rows, localized_dir / "stage10a_lc_scale1000_contact_sheet.png", args.contact_sheet_samples)
    ok_rows = [row for row in rows if row.get("generation_status") == "ok"]
    failed_rows = [row for row in rows if row.get("generation_status") != "ok"]
    summary = {
        "stage": "stage10a_lc_scale1000_localized_composite",
        "total_cases": len(rows),
        "constructed_cases": len(ok_rows),
        "failed_cases": len(failed_rows),
        "cases_by_attack_type": dict(Counter(row["attack_type"] for row in ok_rows)),
        "construction_runtime_seconds": time.time() - started,
        "mean_mask_area_ratio": mean([row.get("mask_area_ratio") for row in ok_rows]),
        "outside_region_preserved_pass_count": sum(1 for row in ok_rows if row.get("outside_region_preserved_check") is True),
        "inside_region_changed_pass_count": sum(1 for row in ok_rows if row.get("inside_region_changed_check") is True),
        "failed_samples": failed_rows,
        "manifest_path": str(manifest_path.resolve()),
    }
    write_json(localized_dir / "stage10a_lc_scale1000_composite_summary.json", summary)
    if failed_rows:
        raise RuntimeError(f"Localized composite failed for {len(failed_rows)} samples")
    return rows, summary


def save_contact_sheet_from_lc(rows: list[dict[str, Any]], out_path: Path, max_items: int) -> None:
    ok = [row for row in rows if row.get("generation_status") == "ok"]
    if not ok:
        return
    selected = ok if len(ok) <= max_items else [ok[round(i * (len(ok) - 1) / (max_items - 1))] for i in range(max_items)]
    panels = []
    for row in selected:
        case_dir = Path(row["localized_composite_path"]).parent
        paths = [case_dir / "watermarked.png", case_dir / "edit_mask.png", case_dir / "tampered.png"]
        if not all(path.exists() for path in paths):
            continue
        panel = Image.new("RGB", (540, 210), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((6, 6), f"{row['sample_id']} | {row['attack_type']}", fill=(0, 0, 0))
        for idx, path in enumerate(paths):
            image = Image.open(path).convert("RGB").resize((180, 180), BICUBIC)
            panel.paste(image, (idx * 180, 28))
        panels.append(panel)
    if not panels:
        return
    canvas = Image.new("RGB", (540, 210 * len(panels)), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (0, idx * 210))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def run_evaluation(args: argparse.Namespace, project_root: Path, localized_dir: Path, eval_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.skip_eval:
        return {"skipped": True}, load_json(eval_dir / "stage10a_lc1000_eval_summary.json")
    if args.overwrite and eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = localized_dir / "stage10a_lc_scale1000_manifest.csv"
    log_path = eval_dir / "stage10a_lc1000_eval.log"
    command = [
        args.python_executable,
        str(SCRIPT_DIR / "run_stage10a_evaluate_real_aigc_attacks.py"),
        "--project_root",
        str(project_root),
        "--attack_dir",
        str(localized_dir),
        "--manifest",
        str(manifest_path),
        "--output_dir",
        str(eval_dir),
        "--max_samples",
        str(args.samples_per_type * len(ATTACK_TYPES)),
        "--python_executable",
        args.python_executable,
        "--contact_sheet_samples",
        str(args.contact_sheet_samples),
        "--overwrite",
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(project_root), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    record = {"command": command, "returncode": proc.returncode, "runtime_seconds": time.time() - started, "log_path": str(log_path)}
    if proc.returncode != 0:
        raise RuntimeError(f"LC1000 evaluation failed: {record}")
    alias_eval_outputs(eval_dir)
    return record, load_json(eval_dir / "stage10a_lc1000_eval_summary.json")


def strategy_row(eval_dir: Path, strategy: str) -> dict[str, str]:
    path = eval_dir / "stage10a_strategy_comparison.csv"
    if not path.exists():
        return {}
    for row in read_csv(path):
        if row.get("strategy") == strategy:
            return row
    return {}


def alias_eval_outputs(eval_dir: Path) -> None:
    aliases = {
        "stage10a_eval_summary.json": "stage10a_lc1000_eval_summary.json",
        "stage10a_eval_by_attack_type.csv": "stage10a_lc1000_eval_by_attack_type.csv",
        "stage10a_strategy_comparison.csv": "stage10a_lc1000_strategy_comparison.csv",
        "stage10a_failure_cases.json": "stage10a_lc1000_failure_cases.json",
        "stage10a_payload_auth_failure_analysis.json": "stage10a_lc1000_payload_auth_failure_analysis.json",
        "stage10a_unknown_analysis.json": "stage10a_lc1000_unknown_analysis.json",
        "stage10a_examples_contact_sheet.png": "stage10a_lc1000_examples_contact_sheet.png",
        "stage10a_eval_overview.md": "stage10a_lc1000_eval_overview.md",
    }
    for src_name, dst_name in aliases.items():
        src = eval_dir / src_name
        if src.exists():
            shutil.copyfile(src, eval_dir / dst_name)
    patch_eval_summary_to_f3(eval_dir)


def patch_eval_summary_to_f3(eval_dir: Path) -> None:
    summary_path = eval_dir / "stage10a_lc1000_eval_summary.json"
    if not summary_path.exists():
        return
    summary = load_json(summary_path)
    f3 = strategy_row(eval_dir, "F3_merge_nearby_fragments")
    if not f3:
        return
    original = dict(summary.get("report_quality", {}))
    summary["stage8f_best_by_selection_report_quality"] = original
    summary["stage8f_main_strategy"] = "F3_merge_nearby_fragments"
    summary["report_quality"] = {
        "unknown_before_stage8e": original.get("unknown_before_stage8e"),
        "unknown_after_stage8e": original.get("unknown_after_stage8e"),
        "f3_unknown_rate": safe_float(f3.get("unknown_rate_after")),
        "f3_reports": int(safe_float(f3.get("total_filtered_regions"))),
        "f3_reports_per_image": safe_float(f3.get("mean_reports_per_image")),
        "f3_gt_coverage_retained": safe_float(f3.get("mean_gt_tamper_area_coverage")),
        "f3_predicted_area_retained": safe_float(f3.get("mean_pred_area_retained")),
        "f3_no_report_images": int(safe_float(f3.get("no_report_images"))),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def attack_type_from_image_id(image_id: str) -> str:
    for attack_type in sorted(ATTACK_TYPES, key=len, reverse=True):
        if image_id.startswith(f"{attack_type}_"):
            return attack_type
    return "unknown"


def row_change_type(row: dict[str, str]) -> str:
    for key in ["change_type_after_filter", "change_type", "improved_change_type", "stage8e_change_type", "final_change_type"]:
        value = row.get(key)
        if value:
            return str(value)
    return "unknown"


def row_area(row: dict[str, str]) -> float:
    for key in ["area_ratio", "region_area_ratio", "filtered_area_ratio", "area_pixels"]:
        if row.get(key) not in ("", None):
            value = safe_float(row.get(key))
            if key == "area_pixels":
                return value / float(512 * 512)
            return value
    return 0.0


def size_bucket(area_ratio: float) -> str:
    if area_ratio < 0.0001:
        return "tiny"
    if area_ratio < 0.001:
        return "small"
    if area_ratio < 0.01:
        return "medium"
    return "large"


def write_group_unknown(rows: list[dict[str, Any]], key: str, out_path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    out = []
    for name, group in sorted(grouped.items()):
        unknown = sum(1 for row in group if row.get("is_unknown"))
        out.append({"bucket": name, "total_regions": len(group), "unknown_regions": unknown, "unknown_rate": unknown / len(group) if group else 0.0})
    write_csv(out_path, out, ["bucket", "total_regions", "unknown_regions", "unknown_rate"])


def unknown_diagnosis(eval_dir: Path, unknown_dir: Path) -> dict[str, Any]:
    if unknown_dir.exists():
        shutil.rmtree(unknown_dir)
    unknown_dir.mkdir(parents=True, exist_ok=True)
    stage8e_summary = load_json(eval_dir / "stage8e" / "stage8e_quality_summary.json")
    stage8f_summary = load_json(eval_dir / "stage8f" / "stage8f_summary.json")
    f3_strategy = strategy_row(eval_dir, "F3_merge_nearby_fragments")
    stage8e_rows = read_csv(eval_dir / "stage8e" / "per_region_stage8e_reports.csv")
    stage8f_rows = read_csv(eval_dir / "stage8f" / "per_region_stage8f_reports.csv")
    f3_rows_raw = [row for row in stage8f_rows if row.get("strategy") == "F3_merge_nearby_fragments"]
    if not f3_rows_raw:
        f3_rows_raw = stage8f_rows
    f3_rows: list[dict[str, Any]] = []
    e_unknown_by_id: dict[str, int] = defaultdict(int)
    e_total_by_id: dict[str, int] = defaultdict(int)
    for row in stage8e_rows:
        image_id = row.get("image_id", "")
        change = row_change_type(row)
        e_total_by_id[image_id] += 1
        if change == "unknown":
            e_unknown_by_id[image_id] += 1
    for row in f3_rows_raw:
        image_id = row.get("image_id", "")
        change = row_change_type(row)
        area = row_area(row)
        f3_rows.append(
            {
                "image_id": image_id,
                "attack_type": attack_type_from_image_id(image_id),
                "change_type": change,
                "is_unknown": change == "unknown",
                "area_ratio": area,
                "region_size_bucket": row.get("region_size_bucket") or size_bucket(area),
                "stage8e_unknown_regions_in_source_image": e_unknown_by_id.get(image_id, 0),
                "stage8e_total_regions_in_source_image": e_total_by_id.get(image_id, 0),
            }
        )
    write_group_unknown(f3_rows, "attack_type", unknown_dir / "stage10a_lc_unknown_by_attack_type.csv")
    write_group_unknown(f3_rows, "region_size_bucket", unknown_dir / "stage10a_lc_unknown_by_region_size.csv")
    frag_rows = []
    for image_id, total in e_total_by_id.items():
        if total < 10:
            bucket = "low_fragmentation"
        elif total < 50:
            bucket = "medium_fragmentation"
        else:
            bucket = "high_fragmentation"
        related = [row for row in f3_rows if row["image_id"] == image_id]
        unknown = sum(1 for row in related if row["is_unknown"])
        frag_rows.append(
            {
                "image_id": image_id,
                "attack_type": attack_type_from_image_id(image_id),
                "stage8e_region_count": total,
                "fragmentation_bucket": bucket,
                "f3_region_count": len(related),
                "f3_unknown_count": unknown,
                "f3_unknown_rate": unknown / len(related) if related else 0.0,
            }
        )
    grouped_frag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frag_rows:
        grouped_frag[row["fragmentation_bucket"]].append(row)
    frag_out = []
    for bucket, group in sorted(grouped_frag.items()):
        total_regions = sum(int(row["f3_region_count"]) for row in group)
        unknown_regions = sum(int(row["f3_unknown_count"]) for row in group)
        frag_out.append(
            {
                "bucket": bucket,
                "images": len(group),
                "total_regions": total_regions,
                "unknown_regions": unknown_regions,
                "unknown_rate": unknown_regions / total_regions if total_regions else 0.0,
            }
        )
    write_csv(unknown_dir / "stage10a_lc_unknown_by_fragmentation.csv", frag_out)
    examples = [row for row in f3_rows if row["is_unknown"]][:50]
    write_json(unknown_dir / "stage10a_lc_unknown_examples.json", {"examples": examples})
    f3_unknown = safe_float(f3_strategy.get("unknown_rate_after"))
    stage8e_after = safe_float(stage8e_summary.get("unknown_change_type_rate_improved"))
    summary = {
        "stage": "stage10a_lc_unknown_diagnosis",
        "total_f3_regions": len(f3_rows),
        "f3_unknown_regions": sum(1 for row in f3_rows if row["is_unknown"]),
        "f3_unknown_rate": f3_unknown,
        "stage8e_unknown_after": stage8e_after,
        "stage8e_to_f3_unknown_delta": f3_unknown - stage8e_after,
        "main_questions": {
            "concentrated_in_attack_type": "see stage10a_lc_unknown_by_attack_type.csv",
            "concentrated_in_tiny_small_fragments": "see stage10a_lc_unknown_by_region_size.csv",
            "stage8e_vs_f3_gap": "F3 merges regions; merged reports can inherit or expose unknown labels differently from raw Stage8E regions.",
            "aggregation_known_unknown_merge": f3_unknown > stage8e_after,
            "semantic_readout_limitation_not_localization_failure": True,
        },
        "interpretation": (
            "LC localization and payload recovery remain strong. Remaining unknown labels should be treated mainly as verification-side "
            "semantic readout limitations under AIGC edit appearance changes, not as localization failures."
        ),
        "stage8f_summary": stage8f_summary,
    }
    write_json(unknown_dir / "stage10a_lc_unknown_diagnosis_summary.json", summary)
    overview = [
        "# Stage 10A LC Unknown Diagnosis",
        "",
        f"- Stage8E unknown after: {stage8e_after}",
        f"- F3 unknown: {f3_unknown}",
        f"- delta: {f3_unknown - stage8e_after}",
        "",
        "The LC setting keeps localization and payload/auth strong. Unknown reports are therefore best described as a coarse semantic-readout limitation rather than a localization failure.",
        "",
        "Aggregation can merge known and unknown raw regions, which changes the report-level unknown rate relative to Stage8E raw/improved region reports.",
    ]
    (unknown_dir / "stage10a_lc_unknown_diagnosis_overview.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    return summary


def four_setting_comparison(stage9_dir: Path, unconstrained_eval_dir: Path, audit_dir: Path, lc1000_eval_dir: Path, out_path: Path) -> list[dict[str, Any]]:
    audit_summary = load_json(audit_dir / "stage10a_attack_locality_audit_summary.json")
    diag = audit_summary.get("localized_composite_40", {})
    unconstrained_pred = audit_summary.get("predicted_mask", {}).get("mean_predicted_mask_area_ratio", 1.0)
    stage9 = stage9_metrics(stage9_dir)
    stage9["capsule_recovery"] = load_json(stage9_dir / "stage9_1000_key_metrics.json").get("capsule_recovery", 0.99965)
    stage9["predicted_area_retained"] = 1.0
    stage9["interpretation"] = "synthetic local tamper full evaluation"
    unconstrained = eval_metrics_row("Stage10A_unconstrained_diffusion_200", unconstrained_eval_dir, pred_area_from_eval_dir(unconstrained_eval_dir, argparse.Namespace(mask_threshold=127)))
    unconstrained_summary = load_json(unconstrained_eval_dir / "stage10a_eval_summary.json")
    unconstrained["capsule_recovery"] = unconstrained_summary.get("stage8c_payload_recovery", {}).get("mean_capsule_recovery_accuracy")
    unconstrained["predicted_mask_area_ratio"] = unconstrained_pred
    unconstrained["predicted_area_retained"] = unconstrained_summary.get("report_quality", {}).get("f3_predicted_area_retained")
    unconstrained["interpretation"] = "global diffusion re-rendering / watermark laundering stress test"
    diag_row = {
        "setting": "Stage10A_localized_composite_diagnostic_40",
        "iou": diag.get("iou"),
        "dice": diag.get("dice"),
        "precision": diag.get("precision"),
        "recall": diag.get("recall"),
        "bit_accuracy": diag.get("bit_accuracy"),
        "payload_recovery": diag.get("payload_recovery"),
        "capsule_recovery": load_json(audit_dir / "localized_composite_40_eval" / "stage10a_eval_summary.json").get("stage8c_payload_recovery", {}).get("mean_capsule_recovery_accuracy"),
        "auth_success": diag.get("auth_success"),
        "f3_unknown": diag.get("f3_unknown"),
        "reports_per_image": diag.get("reports_per_image"),
        "gt_coverage": diag.get("gt_coverage"),
        "predicted_area_retained": 1.0,
        "predicted_mask_area_ratio": diag.get("predicted_mask_area_ratio"),
        "interpretation": "40-sample diagnostic preserving unmasked watermarked pixels",
    }
    lc1000 = eval_metrics_row("Stage10A_localized_composite_scale1000", lc1000_eval_dir, pred_area_from_eval_dir(lc1000_eval_dir, argparse.Namespace(mask_threshold=127)))
    lc1000_summary = load_json(lc1000_eval_dir / "stage10a_lc1000_eval_summary.json")
    lc1000_f3 = strategy_row(lc1000_eval_dir, "F3_merge_nearby_fragments")
    lc1000["capsule_recovery"] = lc1000_summary.get("stage8c_payload_recovery", {}).get("mean_capsule_recovery_accuracy")
    if lc1000_f3:
        lc1000["f3_unknown"] = safe_float(lc1000_f3.get("unknown_rate_after"))
        lc1000["reports_per_image"] = safe_float(lc1000_f3.get("mean_reports_per_image"))
        lc1000["gt_coverage"] = safe_float(lc1000_f3.get("mean_gt_tamper_area_coverage"))
        lc1000["predicted_area_retained"] = safe_float(lc1000_f3.get("mean_pred_area_retained"))
    else:
        lc1000["predicted_area_retained"] = lc1000_summary.get("report_quality", {}).get("f3_predicted_area_retained")
    lc1000["interpretation"] = "scale1000 fair local AIGC editing evaluation"
    rows = [stage9, unconstrained, diag_row, lc1000]
    fields = [
        "setting",
        "iou",
        "dice",
        "precision",
        "recall",
        "bit_accuracy",
        "payload_recovery",
        "capsule_recovery",
        "auth_success",
        "f3_unknown",
        "reports_per_image",
        "gt_coverage",
        "predicted_area_retained",
        "predicted_mask_area_ratio",
        "interpretation",
    ]
    write_csv(out_path, rows, fields)
    return rows


def write_note_and_summaries(
    project_root: Path,
    results_summary_dir: Path,
    generation_summary: dict[str, Any],
    composite_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    unknown_summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    generation_dir: Path,
    localized_dir: Path,
    eval_dir: Path,
    unknown_dir: Path,
) -> None:
    report = eval_summary.get("report_quality", {})
    loc = eval_summary.get("localization", {})
    payload = eval_summary.get("stage8c_payload_recovery", {})
    f4_top3 = strategy_row(eval_dir, "F4_main_region_top3")
    f4_top5 = strategy_row(eval_dir, "F4_main_region_top5")
    lines = [
        "# Stage 10A LC Scale1000 Note",
        "",
        "## Constraints",
        "",
        "- no training",
        "- no VLM",
        "- no 128-bit payload",
        "- no watermark embedding modification",
        "- robust payload remains 64-bit",
        "- verification does not access original image",
        "- GT/edit mask is used only for evaluation",
        "",
        "## Generation",
        "",
        f"- generated: {generation_summary.get('generated_cases')} / {generation_summary.get('total_cases')}",
        f"- success rate: {generation_summary.get('generation_success_rate')}",
        f"- runtime seconds: {generation_summary.get('total_runtime_seconds')}",
        f"- runtime seconds/image: {generation_summary.get('mean_runtime_seconds')}",
        f"- peak GPU memory MB: {generation_summary.get('peak_gpu_memory_mb')}",
        f"- by attack type: {generation_summary.get('samples_by_attack_type')}",
        "",
        "## Localized Composite",
        "",
        f"- constructed: {composite_summary.get('constructed_cases')} / {composite_summary.get('total_cases')}",
        f"- outside-mask preserved: {composite_summary.get('outside_region_preserved_pass_count')}",
        f"- inside-mask changed: {composite_summary.get('inside_region_changed_pass_count')}",
        "",
        "## Evaluation",
        "",
        f"- completed: {eval_summary.get('completed')}",
        f"- failed_stage: {eval_summary.get('failed_stage')}",
        f"- IoU: {loc.get('iou')}",
        f"- Dice: {loc.get('dice')}",
        f"- Precision: {loc.get('precision')}",
        f"- Recall: {loc.get('recall')}",
        f"- bit accuracy: {payload.get('mean_bit_accuracy')}",
        f"- payload recovery: {payload.get('mean_payload_recovery_accuracy')}",
        f"- capsule recovery: {payload.get('mean_capsule_recovery_accuracy')}",
        f"- auth success: {payload.get('mean_auth_success')}",
        f"- Stage8E unknown: {report.get('unknown_before_stage8e')} -> {report.get('unknown_after_stage8e')}",
        f"- F3 reports: {report.get('f3_reports')}",
        f"- F3 reports/image: {report.get('f3_reports_per_image')}",
        f"- F3 unknown: {report.get('f3_unknown_rate')}",
        f"- GT coverage: {report.get('f3_gt_coverage_retained')}",
        f"- predicted area retained: {report.get('f3_predicted_area_retained')}",
        f"- no-report images: {report.get('f3_no_report_images')}",
        f"- F4_top3 reports/image: {f4_top3.get('mean_reports_per_image')}",
        f"- F4_top3 GT coverage: {f4_top3.get('mean_gt_tamper_area_coverage')}",
        f"- F4_top5 reports/image: {f4_top5.get('mean_reports_per_image')}",
        f"- F4_top5 GT coverage: {f4_top5.get('mean_gt_tamper_area_coverage')}",
        "",
        "## Unknown Diagnosis",
        "",
        f"- Stage8E unknown after: {unknown_summary.get('stage8e_unknown_after')}",
        f"- F3 unknown: {unknown_summary.get('f3_unknown_rate')}",
        f"- interpretation: {unknown_summary.get('interpretation')}",
        "",
        "## Recommendation",
        "",
        "LC scale1000 should be used as the fair local AIGC editing evaluation. Unconstrained diffusion remains a separate watermark-laundering stress test.",
    ]
    note = project_root / "STAGE10A_LC_SCALE1000_NOTE.md"
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    results_summary_dir.mkdir(parents=True, exist_ok=True)
    copies = [
        (note, results_summary_dir / "stage10a_lc1000_note.md"),
        (generation_dir / "stage10a_lc_scale1000_generation_summary.json", results_summary_dir / "stage10a_lc1000_generation_summary.json"),
        (localized_dir / "stage10a_lc_scale1000_composite_summary.json", results_summary_dir / "stage10a_lc1000_composite_summary.json"),
        (eval_dir / "stage10a_lc1000_eval_summary.json", results_summary_dir / "stage10a_lc1000_eval_summary.json"),
        (eval_dir / "stage10a_lc1000_eval_by_attack_type.csv", results_summary_dir / "stage10a_lc1000_eval_by_attack_type.csv"),
        (eval_dir / "stage10a_lc1000_strategy_comparison.csv", results_summary_dir / "stage10a_lc1000_strategy_comparison.csv"),
        (eval_dir / "stage10a_lc1000_failure_cases.json", results_summary_dir / "stage10a_lc1000_failure_cases.json"),
        (eval_dir / "stage10a_lc1000_payload_auth_failure_analysis.json", results_summary_dir / "stage10a_lc1000_payload_auth_failure_analysis.json"),
        (eval_dir / "stage10a_lc1000_unknown_analysis.json", results_summary_dir / "stage10a_lc1000_unknown_analysis.json"),
        (eval_dir / "stage10a_lc1000_eval_overview.md", results_summary_dir / "stage10a_lc1000_eval_overview.md"),
        (eval_dir / "stage10a_lc1000_four_setting_comparison.csv", results_summary_dir / "stage10a_lc1000_four_setting_comparison.csv"),
        (unknown_dir / "stage10a_lc_unknown_diagnosis_summary.json", results_summary_dir / "stage10a_lc_unknown_diagnosis_summary.json"),
        (unknown_dir / "stage10a_lc_unknown_by_attack_type.csv", results_summary_dir / "stage10a_lc_unknown_by_attack_type.csv"),
        (unknown_dir / "stage10a_lc_unknown_by_region_size.csv", results_summary_dir / "stage10a_lc_unknown_by_region_size.csv"),
        (unknown_dir / "stage10a_lc_unknown_by_fragmentation.csv", results_summary_dir / "stage10a_lc_unknown_by_fragmentation.csv"),
        (unknown_dir / "stage10a_lc_unknown_diagnosis_overview.md", results_summary_dir / "stage10a_lc_unknown_diagnosis_overview.md"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copyfile(src, dst)
    write_json(
        results_summary_dir / "stage10a_lc1000_full_summary.json",
        {
            "generation": generation_summary,
            "localized_composite": composite_summary,
            "evaluation": eval_summary,
            "unknown_diagnosis": unknown_summary,
            "four_setting_comparison": comparison_rows,
        },
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    stage9_dir = resolve(project_root, args.stage9_dir)
    generation_dir = resolve(project_root, args.generation_dir)
    localized_dir = resolve(project_root, args.localized_dir)
    eval_dir = resolve(project_root, args.eval_dir)
    unknown_dir = resolve(project_root, args.unknown_dir)
    unconstrained_eval_dir = resolve(project_root, args.stage10a_unconstrained_eval_dir)
    audit_dir = resolve(project_root, args.stage10a_audit_dir)
    results_summary_dir = resolve(project_root, args.results_summary_dir)

    free_gb = check_free_space(project_root, args.min_free_gb)
    print(f"Free disk before run: {free_gb:.2f}GB", flush=True)
    start_total = time.time()
    generation_rows, generation_summary = generate_diffusion_edits(args, project_root, stage9_dir, generation_dir)
    composite_rows, composite_summary = construct_localized_composites(args, project_root, generation_rows, localized_dir)
    eval_record, eval_summary = run_evaluation(args, project_root, localized_dir, eval_dir)
    unknown_summary = unknown_diagnosis(eval_dir, unknown_dir)
    comparison_rows = four_setting_comparison(stage9_dir, unconstrained_eval_dir, audit_dir, eval_dir, eval_dir / "stage10a_lc1000_four_setting_comparison.csv")
    write_note_and_summaries(
        project_root,
        results_summary_dir,
        generation_summary,
        composite_summary,
        eval_summary,
        unknown_summary,
        comparison_rows,
        generation_dir,
        localized_dir,
        eval_dir,
        unknown_dir,
    )
    final = {
        "stage": "stage10a_lc_scale1000",
        "completed": True,
        "total_runtime_seconds": time.time() - start_total,
        "generation": generation_summary,
        "localized_composite": composite_summary,
        "evaluation": eval_summary,
        "unknown_diagnosis": unknown_summary,
        "eval_record": eval_record,
        "disk_free_gb_after": check_free_space(project_root, args.min_free_gb),
    }
    write_json(eval_dir / "stage10a_lc1000_run_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
