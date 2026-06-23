from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
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

import generate_editguard_outputs_stage1 as stage1
import run_stage8c_robust_payload_semantic_capsule_audit as s8c


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
NEAREST = getattr(Image, "Resampling", Image).NEAREST

ATTACK_TYPES = ["object_removal", "inpainting", "local_replacement", "local_style_edit"]

PROMPTS = {
    "object_removal": "remove the selected object and naturally fill the background, realistic photo",
    "inpainting": "natural realistic image content, seamless local inpainting, realistic photo",
    "local_replacement": "replace the selected region with a different plausible object or material, realistic photo",
    "local_style_edit": "change only the selected region color, texture, or material style, realistic photo, preserve surrounding context",
}

NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, text, watermark, duplicate objects, artifacts, cartoon, unrealistic, bad anatomy"
)

CSV_FIELDS = [
    "sample_id",
    "source_image_id",
    "original_image_path",
    "watermarked_image_path",
    "attack_type",
    "attack_prompt",
    "negative_prompt",
    "edit_mask_path",
    "generated_tampered_image_path",
    "gt_mask_path",
    "generation_status",
    "generator_name",
    "generator_config",
    "seed",
    "runtime_seconds",
    "gpu_peak_memory_mb",
    "tamper_changed_pixel_ratio",
    "mask_area_ratio",
    "notes",
]


@dataclass(frozen=True)
class AttackCase:
    case_id: str
    source_image_id: str
    attack_type: str
    sample_index: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10A real AIGC editing attack subset generation.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--stage9_dir", default="dfg_locguard/outputs/stage9_full_evaluation_1000")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage10a_real_aigc_attack_subset")
    parser.add_argument("--mode", choices=["smoke", "full", "manifest_only"], default="smoke")
    parser.add_argument("--samples_per_type", type=int, default=50)
    parser.add_argument("--smoke_per_type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--generator", choices=["diffusers_inpaint", "opencv_proxy"], default="diffusers_inpaint")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--model_cache_dir", default="models/hf_cache")
    parser.add_argument("--allow_model_download", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.95)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
        return default if math.isnan(out) else out
    except Exception:
        return default


def successful_stage9_ids(stage9_dir: Path) -> list[dict[str, str]]:
    metrics = read_csv(stage9_dir / "stage8c" / "per_sample_stage8c_metrics.csv")
    rows = [
        row
        for row in metrics
        if row.get("payload_variant") == "P2_id_plus_compact_capsule_64"
        and row.get("status") == "ok"
        and safe_float(row.get("payload_recovery_accuracy")) >= 0.999
        and safe_float(row.get("auth_check_success")) >= 0.999
    ]
    rows.sort(key=lambda row: (safe_float(row.get("fragmentation_ratio")), row.get("image_id", "")))
    return rows


def stratified_cases(stage9_dir: Path, mode: str, samples_per_type: int, smoke_per_type: int, seed: int) -> list[AttackCase]:
    rows = successful_stage9_ids(stage9_dir)
    if not rows:
        raise FileNotFoundError(f"No payload/auth-success Stage 9C rows found in {stage9_dir}")

    per_type = smoke_per_type if mode == "smoke" else samples_per_type
    rng = random.Random(seed)
    # Cover low/mid/high fragmentation by cycling through sorted quantile buckets.
    buckets = [rows[i::4] for i in range(4)]
    for bucket in buckets:
        rng.shuffle(bucket)
    selected: list[AttackCase] = []
    used: set[tuple[str, str]] = set()
    for attack_type in ATTACK_TYPES:
        picked = 0
        cursor = 0
        while picked < per_type:
            bucket = buckets[cursor % len(buckets)]
            if not bucket:
                cursor += 1
                continue
            row = bucket[(cursor // len(buckets)) % len(bucket)]
            image_id = row["image_id"]
            key = (attack_type, image_id)
            cursor += 1
            if key in used:
                continue
            used.add(key)
            case_id = f"{attack_type}_{picked + 1:03d}_{image_id}"
            selected.append(AttackCase(case_id, image_id, attack_type, picked, seed + len(selected) * 17))
            picked += 1
    return selected


def load_binary_mask(path: Path, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size, NEAREST), dtype=np.uint8) > 127


def mask_from_pred(pred_path: Path, attack_type: str, seed: int) -> np.ndarray:
    base = load_binary_mask(pred_path)
    h, w = base.shape
    if int(base.sum()) == 0:
        return fallback_mask(h, w, seed)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(base.astype(np.uint8), 8)
    comps = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area > 0:
            comps.append((area, idx))
    comps.sort(reverse=True)
    keep = np.zeros_like(base)
    max_components = 1 if attack_type in {"object_removal", "local_replacement"} else 2
    for _, idx in comps[:max_components]:
        keep |= labels == idx
    if int(keep.sum()) < 64:
        keep = fallback_mask(h, w, seed)
    kernel_size = 9 if attack_type in {"local_style_edit", "inpainting"} else 17
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    keep = cv2.dilate(keep, kernel, iterations=1).astype(bool)
    return keep


def fallback_mask(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cx = int(w * rng.uniform(0.35, 0.65))
    cy = int(h * rng.uniform(0.35, 0.65))
    rx = int(w * rng.uniform(0.08, 0.16))
    ry = int(h * rng.uniform(0.08, 0.16))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def changed_pixel_ratio(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=2)
    if int(mask.sum()) == 0:
        return 0.0
    return float(np.mean(diff[mask] > 3.0))


def require_diffusers_model(args: argparse.Namespace, project_root: Path) -> None:
    cache_dir = resolve(project_root, args.model_cache_dir)
    model_marker = cache_dir / f"models--{args.model_id.replace('/', '--')}"
    if model_marker.exists() or Path(args.model_id).exists():
        return
    if not args.allow_model_download:
        raise RuntimeError(
            "Diffusers model is not present locally. To download, rerun with --allow_model_download. "
            f"Recommended model: {args.model_id}; estimated disk: 5-7 GB; cache path: {cache_dir}."
        )


def load_diffusers_pipe(args: argparse.Namespace, project_root: Path):
    require_diffusers_model(args, project_root)
    try:
        from diffusers import StableDiffusionInpaintPipeline
    except Exception as exc:
        raise RuntimeError(
            "Missing diffusers inpaint dependencies. Install transformers and accelerate before loading the model."
        ) from exc

    cache_dir = resolve(project_root, args.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        cache_dir=str(cache_dir),
        safety_checker=None,
        requires_safety_checker=False,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_with_diffusers(pipe: Any, image: Image.Image, mask: Image.Image, attack_type: str, seed: int, args: argparse.Namespace) -> Image.Image:
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    result = pipe(
        prompt=PROMPTS[attack_type],
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        mask_image=mask,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        generator=generator,
    ).images[0]
    return result.convert("RGB").resize((512, 512), BICUBIC)


def generate_with_opencv_proxy(image: Image.Image, mask: Image.Image, attack_type: str, seed: int) -> Image.Image:
    # Debug-only proxy. It is not paper evidence for real AIGC editing.
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    m = np.asarray(mask.convert("L"), dtype=np.uint8)
    if attack_type in {"object_removal", "inpainting"}:
        out = cv2.inpaint(arr[:, :, ::-1], m, 7, cv2.INPAINT_TELEA)[:, :, ::-1]
    elif attack_type == "local_replacement":
        rng = np.random.default_rng(seed)
        color = rng.integers(40, 230, size=(3,), dtype=np.uint8)
        out = arr.copy()
        out[m > 127] = (0.3 * out[m > 127] + 0.7 * color).astype(np.uint8)
    else:
        out = arr.copy()
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
        hsv[m > 127, 0] = (hsv[m > 127, 0].astype(np.int32) + 25) % 180
        hsv[m > 127, 1] = np.clip(hsv[m > 127, 1].astype(np.int32) + 30, 0, 255)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(out)


def save_contact_sheet(case_dirs: list[Path], out_path: Path, max_items: int = 50) -> None:
    selected = case_dirs[:max_items] if len(case_dirs) <= max_items else [case_dirs[round(i * (len(case_dirs) - 1) / (max_items - 1))] for i in range(max_items)]
    rows = []
    for case_dir in selected:
        paths = [case_dir / "watermarked.png", case_dir / "edit_mask.png", case_dir / "tampered.png"]
        if not all(path.exists() for path in paths):
            continue
        panels = []
        for path in paths:
            image = Image.open(path).convert("RGB").resize((180, 180), BICUBIC)
            panels.append(image)
        row = Image.new("RGB", (540, 205), "white")
        draw = ImageDraw.Draw(row)
        draw.text((6, 6), case_dir.name, fill=(0, 0, 0))
        for idx, panel in enumerate(panels):
            row.paste(panel, (idx * 180, 25))
        rows.append(row)
    if not rows:
        return
    canvas = Image.new("RGB", (540, 205 * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * 205))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    stage9_dir = resolve(project_root, args.stage9_dir)
    output_dir = resolve(project_root, args.output_dir)
    run_dir = output_dir / ("smoke_8" if args.mode == "smoke" else "full_200")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "generated_attacks").mkdir(parents=True, exist_ok=True)
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    (run_dir / "generation_logs").mkdir(parents=True, exist_ok=True)

    cases = stratified_cases(stage9_dir, args.mode, args.samples_per_type, args.smoke_per_type, args.seed)
    manifest_rows: list[dict[str, Any]] = []
    if args.mode == "manifest_only":
        cases = stratified_cases(stage9_dir, "full", args.samples_per_type, args.smoke_per_type, args.seed)

    model = None
    pipe = None
    if args.mode != "manifest_only":
        model = stage1.load_editguard_model(project_root, project_root / args.opt, project_root / args.ckpt)
        if args.generator == "diffusers_inpaint":
            pipe = load_diffusers_pipe(args, project_root)

    generated_dirs: list[Path] = []
    start_all = time.time()
    for idx, case in enumerate(cases, start=1):
        source_dir = stage9_dir / "stage1_editguard_generated" / case.source_image_id
        original_path = source_dir / "original.png"
        pred_mask_path = source_dir / "editguard_pred_mask.png"
        case_dir = run_dir / "generated_attacks" / case.attack_type / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        edit_mask_path = run_dir / "masks" / f"{case.case_id}_edit_mask.png"
        row = {
            "sample_id": case.case_id,
            "source_image_id": case.source_image_id,
            "original_image_path": str((case_dir / "original.png").resolve()),
            "watermarked_image_path": str((case_dir / "watermarked.png").resolve()),
            "attack_type": case.attack_type,
            "attack_prompt": PROMPTS[case.attack_type],
            "negative_prompt": NEGATIVE_PROMPT,
            "edit_mask_path": str(edit_mask_path.resolve()),
            "generated_tampered_image_path": str((case_dir / "tampered.png").resolve()),
            "gt_mask_path": str((case_dir / "gt_mask.png").resolve()),
            "generation_status": "planned" if args.mode == "manifest_only" else "failed",
            "generator_name": args.generator,
            "generator_config": json.dumps(
                {
                    "model_id": args.model_id if args.generator == "diffusers_inpaint" else "opencv_proxy_debug",
                    "steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "strength": args.strength,
                },
                sort_keys=True,
            ),
            "seed": case.seed,
            "runtime_seconds": "",
            "gpu_peak_memory_mb": "",
            "tamper_changed_pixel_ratio": "",
            "mask_area_ratio": "",
            "notes": "GT mask for Stage10A is the edit mask used to generate this controlled attack; original image is not used during verification.",
        }
        try:
            host = Image.open(original_path).convert("RGB").resize((512, 512), BICUBIC)
            edit_mask = mask_from_pred(pred_mask_path, case.attack_type, case.seed)
            edit_mask_img = Image.fromarray(edit_mask.astype(np.uint8) * 255).convert("L").filter(ImageFilter.MaxFilter(args.mask_dilate))
            edit_mask = np.asarray(edit_mask_img, dtype=np.uint8) > 127
            mask_area_ratio = float(edit_mask.mean())
            Image.fromarray(edit_mask.astype(np.uint8) * 255).save(edit_mask_path)
            if args.mode == "manifest_only":
                row["mask_area_ratio"] = mask_area_ratio
                manifest_rows.append(row)
                continue

            payload = s8c.build_payload_bits(s8c.PayloadVariant("P2_id_plus_compact_capsule_64", 64, "direct", ""), np.asarray(host), case.case_id, idx, args.seed)
            message = s8c.bits_to_message(payload["bits"])
            data = stage1.image_to_editguard_batch(np.asarray(host))
            data["MES"] = message
            assert model is not None
            model.feed_data(data)
            model.mes = message
            watermarked = model.image_hiding()
            watermarked_img = Image.fromarray(watermarked).convert("RGB")

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            start = time.time()
            if args.generator == "diffusers_inpaint":
                assert pipe is not None
                tampered_img = generate_with_diffusers(pipe, watermarked_img, edit_mask_img, case.attack_type, case.seed, args)
            else:
                tampered_img = generate_with_opencv_proxy(watermarked_img, edit_mask_img, case.attack_type, case.seed)
            runtime = time.time() - start
            peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

            host.save(case_dir / "original.png")
            watermarked_img.save(case_dir / "watermarked.png")
            tampered_img.save(case_dir / "tampered.png")
            Image.fromarray(edit_mask.astype(np.uint8) * 255).save(case_dir / "gt_mask.png")
            Image.fromarray(edit_mask.astype(np.uint8) * 255).save(case_dir / "edit_mask.png")
            (case_dir / "payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            (case_dir / "meta.json").write_text(json.dumps({"case": row, "payload_variant": "P2_id_plus_compact_capsule_64"}, ensure_ascii=False, indent=2), encoding="utf-8")

            ratio = changed_pixel_ratio(watermarked, np.asarray(tampered_img), edit_mask)
            row.update(
                {
                    "generation_status": "ok",
                    "runtime_seconds": runtime,
                    "gpu_peak_memory_mb": peak_mb,
                    "tamper_changed_pixel_ratio": ratio,
                    "mask_area_ratio": mask_area_ratio,
                }
            )
            generated_dirs.append(case_dir)
        except Exception as exc:
            row["generation_status"] = "failed"
            row["notes"] = f"{row['notes']} failure={type(exc).__name__}: {exc}"
        manifest_rows.append(row)
        print(f"Stage10A generation {idx}/{len(cases)} {case.case_id}: {row['generation_status']}", flush=True)

    manifest_path = run_dir / "stage10a_attack_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    save_contact_sheet(generated_dirs, run_dir / "stage10a_generation_contact_sheet.png")
    total_runtime = time.time() - start_all
    ok_rows = [row for row in manifest_rows if row["generation_status"] == "ok"]
    summary = {
        "stage": "stage10a_real_aigc_attack_subset_generation",
        "mode": args.mode,
        "generator": args.generator,
        "model_id": args.model_id if args.generator == "diffusers_inpaint" else "opencv_proxy_debug",
        "total_cases": len(manifest_rows),
        "generated_cases": len(ok_rows),
        "failed_cases": len(manifest_rows) - len(ok_rows),
        "generation_success_rate": len(ok_rows) / len(manifest_rows) if manifest_rows else 0.0,
        "samples_by_attack_type": {attack_type: sum(1 for row in ok_rows if row["attack_type"] == attack_type) for attack_type in ATTACK_TYPES},
        "total_runtime_seconds": total_runtime,
        "mean_runtime_seconds": float(np.mean([safe_float(row["runtime_seconds"]) for row in ok_rows])) if ok_rows else 0.0,
        "peak_gpu_memory_mb": max([safe_float(row["gpu_peak_memory_mb"]) for row in ok_rows], default=0.0),
        "manifest_path": str(manifest_path.resolve()),
        "output_dir": str(run_dir.resolve()),
        "constraints": {
            "no_training": True,
            "no_vlm": True,
            "no_128bit_payload": True,
            "watermark_embedding_modified": False,
            "robust_payload_bits": 64,
        },
    }
    (run_dir / "stage10a_generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    overview = [
        "# Stage 10A Real AIGC Attack Generation",
        "",
        f"- mode: {args.mode}",
        f"- generator: {summary['generator']}",
        f"- model_id: {summary['model_id']}",
        f"- generated_cases: {summary['generated_cases']} / {summary['total_cases']}",
        f"- success_rate: {summary['generation_success_rate']}",
        f"- peak_gpu_memory_mb: {summary['peak_gpu_memory_mb']}",
        "",
        "No training, VLM, 128-bit payload, or watermark embedding modification is used.",
    ]
    (run_dir / "stage10a_generation_overview.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
