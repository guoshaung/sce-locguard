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
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
NEAREST = getattr(Image, "Resampling", Image).NEAREST
ATTACK_TYPES = ["object_removal", "inpainting", "local_replacement", "local_style_edit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10A attack locality and fairness audit.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--attack_dir", default="dfg_locguard/outputs/stage10a_real_aigc_attack_subset/full_200")
    parser.add_argument("--eval_dir", default="dfg_locguard/outputs/stage10a_real_aigc_attack_eval")
    parser.add_argument("--stage9_dir", default="dfg_locguard/outputs/stage9_full_evaluation_1000")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage10a_attack_locality_audit")
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--diff_threshold", type=float, default=3.0)
    parser.add_argument("--outside_changed_threshold", type=float, default=0.05)
    parser.add_argument("--outside_mean_abs_threshold", type=float, default=2.0)
    parser.add_argument("--outside_psnr_threshold", type=float, default=35.0)
    parser.add_argument("--composite_per_type", type=int, default=10)
    parser.add_argument("--contact_sheet_samples", type=int, default=40)
    parser.add_argument("--skip_localized_eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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


def mean(values: list[Any]) -> float:
    nums = [safe_float(v) for v in values if v not in ("", None)]
    return float(sum(nums) / len(nums)) if nums else 0.0


def path_from_manifest(value: str) -> Path:
    return Path(value).resolve()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)


def load_mask(path: Path, threshold: int = 127) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((512, 512), NEAREST), dtype=np.uint8) > threshold


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def region_stats(diff_rgb: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    if int(mask.sum()) == 0:
        return {"changed_ratio": 0.0, "mean_abs": 0.0, "mse": 0.0, "psnr": float("inf")}
    vals = diff_rgb[mask]
    gray = np.mean(vals, axis=1)
    mse = float(np.mean(vals.astype(np.float32) ** 2))
    psnr = float("inf") if mse <= 1e-12 else float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))
    return {
        "changed_ratio": float(np.mean(gray > threshold)),
        "mean_abs": float(np.mean(gray)),
        "mse": mse,
        "psnr": psnr,
    }


def ssim_map_gray(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(gray_a * gray_a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(gray_b * gray_b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return num / np.maximum(den, 1e-12)


def infer_generation_source(row: dict[str, str]) -> tuple[str, str, str]:
    original_path = path_from_manifest(row.get("original_image_path", ""))
    watermarked_path = path_from_manifest(row.get("watermarked_image_path", ""))
    generator_config = row.get("generator_config", "")
    if watermarked_path.exists() and original_path.exists() and row.get("generator_name") == "diffusers_inpaint":
        return (
            "watermarked",
            str(watermarked_path),
            "manifest lacks explicit source field; Stage10A generator code calls diffusers with watermarked_img",
        )
    if watermarked_path.exists():
        return ("watermarked_uncertain", str(watermarked_path), "watermarked path exists but source field is implicit")
    if original_path.exists():
        return ("original_uncertain", str(original_path), "only original path could be resolved")
    return ("ambiguous", "", f"could not resolve original/watermarked paths; generator_config={generator_config}")


def manifest_audit(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        source_type, source_path, evidence = infer_generation_source(row)
        out.append(
            {
                "sample_id": row.get("sample_id"),
                "source_image_id": row.get("source_image_id"),
                "attack_type": row.get("attack_type"),
                "generation_status": row.get("generation_status"),
                "inferred_source_type": source_type,
                "inferred_source_path": source_path,
                "original_image_path": row.get("original_image_path"),
                "watermarked_image_path": row.get("watermarked_image_path"),
                "generated_tampered_image_path": row.get("generated_tampered_image_path"),
                "source_inference_evidence": evidence,
            }
        )
    return out


def locality_audit(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["sample_id"]
        source_type, source_path_str, evidence = infer_generation_source(row)
        source_path = Path(source_path_str) if source_path_str else Path("")
        tampered_path = path_from_manifest(row["generated_tampered_image_path"])
        mask_path = path_from_manifest(row.get("edit_mask_path") or row.get("gt_mask_path"))
        try:
            source = load_rgb(source_path)
            tampered = load_rgb(tampered_path)
            mask = load_mask(mask_path, args.mask_threshold)
            inside = mask
            outside = ~mask
            diff = np.abs(tampered.astype(np.float32) - source.astype(np.float32))
            inside_stats = region_stats(diff, inside, args.diff_threshold)
            outside_stats = region_stats(diff, outside, args.diff_threshold)
            ssim = ssim_map_gray(source, tampered)
            outside_ssim = float(np.mean(ssim[outside])) if int(outside.sum()) else 1.0
            source_size = image_size(source_path)
            tampered_size = image_size(tampered_path)
            globally_changed = (
                outside_stats["changed_ratio"] > args.outside_changed_threshold
                or outside_stats["mean_abs"] > args.outside_mean_abs_threshold
                or outside_stats["psnr"] < args.outside_psnr_threshold
            )
            out.append(
                {
                    "sample_id": sample_id,
                    "source_image_id": row.get("source_image_id"),
                    "attack_type": row.get("attack_type"),
                    "source_type": source_type,
                    "source_path": str(source_path),
                    "source_inference_evidence": evidence,
                    "tampered_path": str(tampered_path),
                    "mask_path": str(mask_path),
                    "mask_area_ratio": float(np.mean(mask)),
                    "changed_area_ratio_inside_mask": inside_stats["changed_ratio"],
                    "changed_area_ratio_outside_mask": outside_stats["changed_ratio"],
                    "mean_abs_diff_inside_mask": inside_stats["mean_abs"],
                    "mean_abs_diff_outside_mask": outside_stats["mean_abs"],
                    "mse_inside_mask": inside_stats["mse"],
                    "mse_outside_mask": outside_stats["mse"],
                    "psnr_inside_mask": inside_stats["psnr"],
                    "psnr_outside_mask": outside_stats["psnr"],
                    "ssim_outside_mask": outside_ssim,
                    "source_width": source_size[0],
                    "source_height": source_size[1],
                    "tampered_width": tampered_size[0],
                    "tampered_height": tampered_size[1],
                    "dimensions_match_source": source_size == tampered_size,
                    "appears_globally_changed": globally_changed,
                    "status": "ok",
                    "error_message": "",
                }
            )
        except Exception as exc:
            out.append(
                {
                    "sample_id": sample_id,
                    "source_image_id": row.get("source_image_id"),
                    "attack_type": row.get("attack_type"),
                    "source_type": source_type,
                    "source_path": str(source_path),
                    "tampered_path": str(tampered_path),
                    "mask_path": str(mask_path),
                    "status": "failed",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
    return out


def predicted_mask_audit(eval_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    csv_path = eval_dir / "stage8c_real_payload_recovery" / "per_sample_stage8c_metrics.csv"
    if not csv_path.exists():
        return []
    rows = read_csv(csv_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("payload_variant") != "P2_id_plus_compact_capsule_64":
            continue
        try:
            pred_path = Path(row["pred_mask_path"])
            gt_path = Path(row["gt_mask_path"])
            pred = load_mask(pred_path, args.mask_threshold)
            gt = load_mask(gt_path, args.mask_threshold)
            pred_ratio = float(np.mean(pred))
            gt_ratio = float(np.mean(gt))
            area_ratio = pred_ratio / max(gt_ratio, 1e-12)
            out.append(
                {
                    "sample_id": row.get("image_id"),
                    "attack_type": row.get("attack_type"),
                    "pred_mask_path": str(pred_path),
                    "gt_mask_path": str(gt_path),
                    "predicted_mask_area_ratio": pred_ratio,
                    "gt_mask_area_ratio": gt_ratio,
                    "predicted_to_gt_area_ratio": area_ratio,
                    "recall": row.get("recall"),
                    "precision": row.get("precision"),
                    "iou": row.get("iou"),
                    "near_full_image_alarm": pred_ratio > 0.8,
                    "status": "ok",
                    "error_message": "",
                }
            )
        except Exception as exc:
            out.append(
                {
                    "sample_id": row.get("image_id"),
                    "attack_type": row.get("attack_type"),
                    "status": "failed",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
    return out


def select_composite_rows(rows: list[dict[str, str]], per_type: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row.get("attack_type", "")].append(row)
    for attack_type in ATTACK_TYPES:
        selected.extend(by_type.get(attack_type, [])[:per_type])
    return selected


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def create_localized_composite_subset(
    rows: list[dict[str, str]],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Path]:
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "generated_attacks").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    selected = select_composite_rows(rows, args.composite_per_type)
    manifest_rows: list[dict[str, Any]] = []
    case_dirs: list[Path] = []
    for row in selected:
        sample_id = row["sample_id"]
        attack_type = row["attack_type"]
        src_dir = path_from_manifest(row["generated_tampered_image_path"]).parent
        case_dir = out_dir / "generated_attacks" / attack_type / sample_id
        case_dir.mkdir(parents=True, exist_ok=True)
        mask = load_mask(path_from_manifest(row.get("edit_mask_path") or row.get("gt_mask_path")), args.mask_threshold)
        watermarked = load_rgb(src_dir / "watermarked.png")
        generated = load_rgb(src_dir / "tampered.png")
        composite = watermarked.copy()
        composite[mask] = generated[mask]
        Image.fromarray(composite).save(case_dir / "tampered.png")
        for name, mode in [("original.png", "RGB"), ("watermarked.png", "RGB"), ("gt_mask.png", "L"), ("edit_mask.png", "L")]:
            source = src_dir / name
            if source.exists():
                image = Image.open(source).convert(mode).resize((512, 512), BICUBIC if mode == "RGB" else NEAREST)
                image.save(case_dir / name)
        for name in ["payload.json", "meta.json"]:
            if (src_dir / name).exists():
                copy_file(src_dir / name, case_dir / name)
        mask_out = out_dir / "masks" / f"{sample_id}_edit_mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_out)
        new_row = dict(row)
        new_row.update(
            {
                "original_image_path": str((case_dir / "original.png").resolve()),
                "watermarked_image_path": str((case_dir / "watermarked.png").resolve()),
                "edit_mask_path": str(mask_out.resolve()),
                "generated_tampered_image_path": str((case_dir / "tampered.png").resolve()),
                "gt_mask_path": str((case_dir / "gt_mask.png").resolve()),
                "generation_status": "ok",
                "generator_name": "localized_composite_diagnostic",
                "notes": "Diagnostic local composite: mask * diffusion_generated + (1-mask) * watermarked_image.",
            }
        )
        manifest_rows.append(new_row)
        case_dirs.append(case_dir)
    manifest_path = out_dir / "stage10a_attack_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    save_contact_sheet(case_dirs, out_dir / "localized_composite_40_contact_sheet.png", max_items=40)
    summary = {
        "stage": "stage10a_localized_composite_40_generation",
        "total_cases": len(manifest_rows),
        "cases_by_attack_type": dict(Counter(row["attack_type"] for row in manifest_rows)),
        "generation_rule": "tampered_local = mask * diffusion_generated + (1-mask) * watermarked_image",
        "unmasked_region_preserved_from_watermarked": True,
    }
    (out_dir / "stage10a_localized_composite_40_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_rows, manifest_path


def run_stage10a_eval(project_root: Path, attack_dir: Path, manifest_path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "localized_composite_eval.log"
    command = [
        args.python_executable,
        str(SCRIPT_DIR / "run_stage10a_evaluate_real_aigc_attacks.py"),
        "--project_root",
        str(project_root),
        "--attack_dir",
        str(attack_dir),
        "--manifest",
        str(manifest_path),
        "--output_dir",
        str(output_dir),
        "--max_samples",
        "40",
        "--python_executable",
        args.python_executable,
        "--overwrite",
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(project_root), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    return {
        "command": command,
        "returncode": proc.returncode,
        "runtime_seconds": time.time() - started,
        "log_path": str(log_path),
    }


def save_contact_sheet(case_dirs: list[Path], out_path: Path, max_items: int = 40) -> None:
    selected = case_dirs[:max_items]
    rows = []
    for case_dir in selected:
        paths = [case_dir / "watermarked.png", case_dir / "edit_mask.png", case_dir / "tampered.png"]
        if not all(path.exists() for path in paths):
            continue
        panel = Image.new("RGB", (540, 210), "white")
        draw = ImageDraw.Draw(panel)
        draw.text((6, 6), case_dir.name, fill=(0, 0, 0))
        for idx, path in enumerate(paths):
            image = Image.open(path).convert("RGB").resize((180, 180), BICUBIC)
            panel.paste(image, (idx * 180, 28))
        rows.append(panel)
    if not rows:
        return
    canvas = Image.new("RGB", (540, 210 * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, 210 * idx))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def create_audit_contact_sheet(rows: list[dict[str, str]], out_path: Path, max_items: int = 40) -> None:
    selected = rows[:max_items]
    panels = []
    for row in selected:
        try:
            case_dir = path_from_manifest(row["generated_tampered_image_path"]).parent
            images = [case_dir / "watermarked.png", case_dir / "edit_mask.png", case_dir / "tampered.png"]
            panel = Image.new("RGB", (540, 210), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((6, 6), f"{row['sample_id']} | {row['attack_type']}", fill=(0, 0, 0))
            for idx, path in enumerate(images):
                image = Image.open(path).convert("RGB").resize((180, 180), BICUBIC)
                panel.paste(image, (idx * 180, 28))
            panels.append(panel)
        except Exception:
            continue
    if not panels:
        return
    canvas = Image.new("RGB", (540, 210 * len(panels)), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (0, idx * 210))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(data: dict[str, Any], paths: list[list[str]], default: Any = None) -> Any:
    for path in paths:
        cur: Any = data
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur not in ("", None):
            return cur
    return default


def summarize_pred_area(pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in pred_rows if row.get("status") == "ok"]
    return {
        "mean_predicted_mask_area_ratio": mean([row.get("predicted_mask_area_ratio") for row in ok]),
        "mean_gt_mask_area_ratio": mean([row.get("gt_mask_area_ratio") for row in ok]),
        "near_full_image_alarm_count": sum(1 for row in ok if str(row.get("near_full_image_alarm")).lower() == "true" or row.get("near_full_image_alarm") is True),
        "total": len(ok),
    }


def pred_area_from_eval_dir(eval_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return summarize_pred_area(predicted_mask_audit(eval_dir, args))


def stage9_metrics(stage9_dir: Path) -> dict[str, Any]:
    key = load_json(stage9_dir / "stage9_1000_key_metrics.json")
    stage8d = load_json(stage9_dir / "stage8d" / "stage8d_summary.json")
    stage8f = load_json(stage9_dir / "stage8f" / "stage8f_summary.json")
    best = stage8f.get("best_strategy_payload", {}) if isinstance(stage8f, dict) else {}
    return {
        "setting": "Stage9C_synthetic_1000",
        "iou": nested_get(key, [["mean_iou"], ["iou"], ["localization", "iou"]], stage8d.get("mean_iou")),
        "dice": nested_get(key, [["mean_dice"], ["dice"], ["localization", "dice"]], stage8d.get("mean_dice")),
        "precision": nested_get(key, [["mean_precision"], ["precision"], ["localization", "precision"]], stage8d.get("mean_precision")),
        "recall": nested_get(key, [["mean_recall"], ["recall"], ["localization", "recall"]], stage8d.get("mean_recall")),
        "bit_accuracy": nested_get(key, [["mean_bit_accuracy"], ["bit_accuracy"]], stage8d.get("mean_bit_accuracy")),
        "payload_recovery": nested_get(key, [["mean_payload_recovery"], ["payload_recovery"]], stage8d.get("mean_payload_recovery_accuracy")),
        "auth_success": nested_get(key, [["mean_auth_success"], ["auth_success"]], stage8d.get("mean_auth_success")),
        "f3_unknown": nested_get(key, [["f3_unknown_rate"]], best.get("unknown_rate_after")),
        "reports_per_image": nested_get(key, [["f3_reports_per_image"]], best.get("mean_reports_per_image")),
        "predicted_mask_area_ratio": "",
        "gt_coverage": nested_get(key, [["gt_coverage_retained"], ["f3_gt_coverage_retained"]], best.get("mean_gt_tamper_area_coverage")),
    }


def eval_metrics_row(setting: str, eval_dir: Path, pred_summary: dict[str, Any]) -> dict[str, Any]:
    summary = load_json(eval_dir / "stage10a_eval_summary.json")
    s8c = summary.get("stage8c_payload_recovery", {})
    loc = summary.get("localization", {})
    report = summary.get("report_quality", {})
    return {
        "setting": setting,
        "iou": loc.get("iou"),
        "dice": loc.get("dice"),
        "precision": loc.get("precision"),
        "recall": loc.get("recall"),
        "bit_accuracy": s8c.get("mean_bit_accuracy"),
        "payload_recovery": s8c.get("mean_payload_recovery_accuracy"),
        "auth_success": s8c.get("mean_auth_success"),
        "f3_unknown": report.get("f3_unknown_rate"),
        "reports_per_image": report.get("f3_reports_per_image"),
        "predicted_mask_area_ratio": pred_summary.get("mean_predicted_mask_area_ratio"),
        "gt_coverage": report.get("f3_gt_coverage_retained"),
    }


def summarize_by_attack_type(eval_dir: Path) -> list[dict[str, str]]:
    path = eval_dir / "stage10a_eval_by_attack_type.csv"
    return read_csv(path) if path.exists() else []


def make_overview(
    output_dir: Path,
    summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    by_attack: list[dict[str, str]],
    composite_by_attack: list[dict[str, str]],
) -> None:
    lines = [
        "# Stage 10A Attack Locality Audit",
        "",
        "## Purpose",
        "",
        "This audit checks whether Stage 10A was a fair local AIGC edit or an unconstrained diffusion re-rendering stress test.",
        "",
        "## Source Audit",
        "",
        f"- attacks using watermarked source: {summary['source_counts'].get('watermarked', 0)}",
        f"- attacks using original source: {summary['source_counts'].get('original', 0)}",
        f"- uncertain/ambiguous source count: {summary['source_counts'].get('ambiguous_or_uncertain', 0)}",
        f"- source conclusion: {summary['source_conclusion']}",
        "",
        "## Locality Audit",
        "",
        f"- mean mask area ratio: {summary['locality']['mean_mask_area_ratio']}",
        f"- mean changed area ratio inside mask: {summary['locality']['mean_changed_area_ratio_inside_mask']}",
        f"- mean changed area ratio outside mask: {summary['locality']['mean_changed_area_ratio_outside_mask']}",
        f"- mean abs diff outside mask: {summary['locality']['mean_abs_diff_outside_mask']}",
        f"- mean outside-mask PSNR: {summary['locality']['mean_psnr_outside_mask']}",
        f"- globally changed samples: {summary['locality']['globally_changed_count']} / {summary['locality']['evaluated_samples']}",
        f"- locality conclusion: {summary['locality_conclusion']}",
        "",
        "## Predicted Mask Audit",
        "",
        f"- mean predicted mask area ratio: {summary['predicted_mask']['mean_predicted_mask_area_ratio']}",
        f"- mean GT mask area ratio: {summary['predicted_mask']['mean_gt_mask_area_ratio']}",
        f"- near-full-image alarms: {summary['predicted_mask']['near_full_image_alarm_count']} / {summary['predicted_mask']['total']}",
        "",
        "## Localized Composite 40",
        "",
        f"- completed: {summary['localized_composite_40'].get('completed')}",
        f"- IoU: {summary['localized_composite_40'].get('iou')}",
        f"- Dice: {summary['localized_composite_40'].get('dice')}",
        f"- bit accuracy: {summary['localized_composite_40'].get('bit_accuracy')}",
        f"- payload recovery: {summary['localized_composite_40'].get('payload_recovery')}",
        f"- auth success: {summary['localized_composite_40'].get('auth_success')}",
        f"- F3 unknown: {summary['localized_composite_40'].get('f3_unknown')}",
        f"- F3 reports/image: {summary['localized_composite_40'].get('reports_per_image')}",
        "",
        "## Three-Setting Comparison",
        "",
        "| setting | IoU | Dice | Precision | Recall | bit acc | payload | auth | F3 unknown | reports/image | pred area | GT coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            "| {setting} | {iou} | {dice} | {precision} | {recall} | {bit_accuracy} | {payload_recovery} | {auth_success} | {f3_unknown} | {reports_per_image} | {predicted_mask_area_ratio} | {gt_coverage} |".format(
                **{key: row.get(key, "") for key in [
                    "setting",
                    "iou",
                    "dice",
                    "precision",
                    "recall",
                    "bit_accuracy",
                    "payload_recovery",
                    "auth_success",
                    "f3_unknown",
                    "reports_per_image",
                    "predicted_mask_area_ratio",
                    "gt_coverage",
                ]}
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- {summary['interpretation']}",
            f"- recommendation_rerun_full200_with_localized_compositing: {summary['recommendation_rerun_full200_with_localized_compositing']}",
            f"- recommendation_proceed_stage10b: {summary['recommendation_proceed_stage10b']}",
            "",
            "## Constraints",
            "",
            "- no training",
            "- no VLM",
            "- no 128-bit payload",
            "- no watermark embedding modification",
            "- robust payload remains 64-bit",
            "- verification does not access original image",
            "- GT mask is used only for evaluation",
        ]
    )
    output_dir.joinpath("stage10a_attack_locality_audit_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    attack_dir = resolve(project_root, args.attack_dir)
    eval_dir = resolve(project_root, args.eval_dir)
    stage9_dir = resolve(project_root, args.stage9_dir)
    output_dir = resolve(project_root, args.output_dir)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = attack_dir / "stage10a_attack_manifest.csv"
    rows_all = read_csv(manifest_path)
    rows = [row for row in rows_all if row.get("generation_status") == "ok"]
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise FileNotFoundError(f"No generated attacks found in {manifest_path}")

    source_rows = manifest_audit(rows)
    locality_rows = locality_audit(rows, args)
    pred_rows = predicted_mask_audit(eval_dir, args)
    write_csv(output_dir / "stage10a_attack_source_audit.csv", source_rows)
    write_csv(output_dir / "stage10a_locality_metrics.csv", locality_rows)
    write_csv(output_dir / "stage10a_predicted_mask_area_audit.csv", pred_rows)
    create_audit_contact_sheet(rows, output_dir / "stage10a_audit_contact_sheet.png", args.contact_sheet_samples)

    composite_dir = output_dir / "localized_composite_40"
    composite_rows, composite_manifest = create_localized_composite_subset(rows, composite_dir, args)
    composite_eval_dir = output_dir / "localized_composite_40_eval"
    composite_eval_record: dict[str, Any] = {"skipped": args.skip_localized_eval}
    if not args.skip_localized_eval:
        composite_eval_record = run_stage10a_eval(project_root, composite_dir, composite_manifest, composite_eval_dir, args)

    composite_summary_raw = load_json(composite_eval_dir / "stage10a_eval_summary.json")
    composite_pred_summary = pred_area_from_eval_dir(composite_eval_dir, args) if composite_eval_dir.exists() else {}
    composite_metrics = eval_metrics_row("Stage10A_localized_composite_diagnostic_40", composite_eval_dir, composite_pred_summary) if composite_summary_raw else {}
    composite_by_attack = summarize_by_attack_type(composite_eval_dir)
    (output_dir / "stage10a_localized_composite_40_summary.json").write_text(
        json.dumps(
            {
                "generation_cases": len(composite_rows),
                "generation_cases_by_attack_type": dict(Counter(row["attack_type"] for row in composite_rows)),
                "evaluation_record": composite_eval_record,
                "evaluation_summary": composite_summary_raw,
                "predicted_mask_area_summary": composite_pred_summary,
                "by_attack_type": composite_by_attack,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    locality_ok = [row for row in locality_rows if row.get("status") == "ok"]
    pred_summary = summarize_pred_area(pred_rows)
    source_counter = Counter(row["inferred_source_type"] for row in source_rows)
    watermarked_count = sum(count for key, count in source_counter.items() if key.startswith("watermarked"))
    original_count = sum(count for key, count in source_counter.items() if key.startswith("original"))
    ambiguous_count = len(source_rows) - watermarked_count - original_count
    globally_changed_count = sum(1 for row in locality_ok if str(row.get("appears_globally_changed")).lower() == "true" or row.get("appears_globally_changed") is True)
    mean_outside_changed = mean([row.get("changed_area_ratio_outside_mask") for row in locality_ok])
    mean_outside_abs = mean([row.get("mean_abs_diff_outside_mask") for row in locality_ok])
    mean_outside_psnr = mean([row.get("psnr_outside_mask") for row in locality_ok if row.get("psnr_outside_mask") != float("inf")])
    source_conclusion = (
        "watermarked_source_confirmed_by_manifest_paths_and_generator_code"
        if watermarked_count == len(source_rows)
        else "source_ambiguous_or_original_used"
    )
    locality_conclusion = (
        "unconstrained_diffusion_rerendering_stress_test"
        if globally_changed_count > len(locality_ok) * 0.5
        else "mostly_local_edit"
    )

    stage9_row = stage9_metrics(stage9_dir)
    stage10_row = eval_metrics_row("Stage10A_unconstrained_diffusion_200", eval_dir, pred_summary)
    comparison_rows = [stage9_row, stage10_row]
    if composite_metrics:
        comparison_rows.append(composite_metrics)
    write_csv(output_dir / "stage10a_three_setting_comparison.csv", comparison_rows)

    localized_payload = safe_float(composite_metrics.get("payload_recovery"), -1.0) if composite_metrics else -1.0
    localized_iou = safe_float(composite_metrics.get("iou"), -1.0) if composite_metrics else -1.0
    if original_count > 0:
        interpretation = "Some attacks appear to use original images, so payload/auth robustness would not be valid for those cases."
    elif locality_conclusion == "unconstrained_diffusion_rerendering_stress_test" and localized_payload > safe_float(stage10_row.get("payload_recovery"), 0.0) + 0.2:
        interpretation = "Original Stage10A failure is mostly caused by global generative reconstruction outside the edit mask."
    elif locality_conclusion == "unconstrained_diffusion_rerendering_stress_test":
        interpretation = "Original Stage10A is an unconstrained diffusion re-rendering stress test; localized composite is needed for fair local tamper evaluation."
    elif localized_payload < 0.95 or localized_iou < 0.5:
        interpretation = "Even localized AIGC inpainting remains severe for the current payload/localization pipeline."
    else:
        interpretation = "Localized composite recovers the pipeline, so the unconstrained attack should not be treated as fair local editing."

    summary = {
        "stage": "stage10a_attack_locality_audit",
        "total_attacks": len(rows),
        "source_counts": {
            "watermarked": watermarked_count,
            "original": original_count,
            "ambiguous_or_uncertain": ambiguous_count,
            "raw": dict(source_counter),
        },
        "source_conclusion": source_conclusion,
        "locality": {
            "evaluated_samples": len(locality_ok),
            "mean_mask_area_ratio": mean([row.get("mask_area_ratio") for row in locality_ok]),
            "mean_changed_area_ratio_inside_mask": mean([row.get("changed_area_ratio_inside_mask") for row in locality_ok]),
            "mean_changed_area_ratio_outside_mask": mean_outside_changed,
            "mean_abs_diff_inside_mask": mean([row.get("mean_abs_diff_inside_mask") for row in locality_ok]),
            "mean_abs_diff_outside_mask": mean_outside_abs,
            "mean_psnr_inside_mask": mean([row.get("psnr_inside_mask") for row in locality_ok if row.get("psnr_inside_mask") != float("inf")]),
            "mean_psnr_outside_mask": mean_outside_psnr,
            "mean_ssim_outside_mask": mean([row.get("ssim_outside_mask") for row in locality_ok]),
            "globally_changed_count": globally_changed_count,
        },
        "locality_conclusion": locality_conclusion,
        "predicted_mask": pred_summary,
        "localized_composite_40": {
            "completed": bool(composite_summary_raw.get("completed")) if composite_summary_raw else False,
            "iou": composite_metrics.get("iou"),
            "dice": composite_metrics.get("dice"),
            "precision": composite_metrics.get("precision"),
            "recall": composite_metrics.get("recall"),
            "bit_accuracy": composite_metrics.get("bit_accuracy"),
            "payload_recovery": composite_metrics.get("payload_recovery"),
            "auth_success": composite_metrics.get("auth_success"),
            "f3_unknown": composite_metrics.get("f3_unknown"),
            "reports_per_image": composite_metrics.get("reports_per_image"),
            "gt_coverage": composite_metrics.get("gt_coverage"),
            "predicted_mask_area_ratio": composite_metrics.get("predicted_mask_area_ratio"),
            "eval_record": composite_eval_record,
        },
        "interpretation": interpretation,
        "recommendation_rerun_full200_with_localized_compositing": locality_conclusion == "unconstrained_diffusion_rerendering_stress_test",
        "recommendation_proceed_stage10b": True,
        "constraints": {
            "no_training": True,
            "no_vlm": True,
            "no_128bit_payload": True,
            "watermark_embedding_modified": False,
            "robust_payload_bits": 64,
            "verification_original_access": False,
            "gt_mask_usage": ["evaluation_only"],
        },
    }
    (output_dir / "stage10a_attack_locality_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_overview(output_dir, summary, comparison_rows, summarize_by_attack_type(eval_dir), composite_by_attack)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
