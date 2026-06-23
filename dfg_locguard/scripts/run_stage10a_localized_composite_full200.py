from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_stage10a_attack_locality_audit import (  # noqa: E402
    ATTACK_TYPES,
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


MANIFEST_FIELDS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10A localized-composite full 200 construction and evaluation.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--source_attack_dir", default="dfg_locguard/outputs/stage10a_real_aigc_attack_subset/full_200")
    parser.add_argument("--source_eval_dir", default="dfg_locguard/outputs/stage10a_real_aigc_attack_eval")
    parser.add_argument("--audit_dir", default="dfg_locguard/outputs/stage10a_attack_locality_audit")
    parser.add_argument("--stage9_dir", default="dfg_locguard/outputs/stage9_full_evaluation_1000")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage10a_localized_composite_full_200")
    parser.add_argument("--eval_output_dir", default="dfg_locguard/outputs/stage10a_localized_composite_full_200_eval")
    parser.add_argument("--results_summary_dir", default="dfg_locguard/results_summary")
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)


def load_mask(path: Path, threshold: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((512, 512), NEAREST), dtype=np.uint8) > threshold


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def mean(values: list[Any]) -> float:
    nums = [safe_float(v) for v in values if v not in ("", None)]
    return float(sum(nums) / len(nums)) if nums else 0.0


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def case_dir_from_row(row: dict[str, str]) -> Path:
    return Path(row["generated_tampered_image_path"]).resolve().parent


def build_localized_row(row: dict[str, str], output_dir: Path, mask_threshold: int) -> dict[str, Any]:
    sample_id = row["sample_id"]
    attack_type = row["attack_type"]
    src_dir = case_dir_from_row(row)
    out_dir = output_dir / "localized_attacks" / attack_type / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    watermarked_path = src_dir / "watermarked.png"
    diffusion_path = src_dir / "tampered.png"
    original_path = src_dir / "original.png"
    gt_mask_path = src_dir / "gt_mask.png"
    edit_mask_src = Path(row.get("edit_mask_path") or gt_mask_path).resolve()
    edit_mask_out = output_dir / "masks" / f"{sample_id}_edit_mask.png"
    composite_path = out_dir / "tampered.png"

    out_row: dict[str, Any] = {
        "sample_id": sample_id,
        "source_image_id": row.get("source_image_id", ""),
        "attack_type": attack_type,
        "watermarked_source_path": str((out_dir / "watermarked.png").resolve()),
        "diffusion_generated_path": str(diffusion_path),
        "edit_mask_path": str(edit_mask_out.resolve()),
        "localized_composite_path": str(composite_path.resolve()),
        "gt_mask_path": str((out_dir / "gt_mask.png").resolve()),
        "generation_status": "failed",
        "error_message": "",
        "original_image_path": str((out_dir / "original.png").resolve()),
        "watermarked_image_path": str((out_dir / "watermarked.png").resolve()),
        "generated_tampered_image_path": str(composite_path.resolve()),
        "attack_prompt": row.get("attack_prompt", ""),
        "negative_prompt": row.get("negative_prompt", ""),
        "generator_name": "localized_composite_diagnostic_full200",
        "generator_config": json.dumps(
            {"rule": "tampered_local = mask * diffusion_generated + (1-mask) * watermarked_image"},
            sort_keys=True,
        ),
        "seed": row.get("seed", ""),
        "runtime_seconds": "",
        "gpu_peak_memory_mb": "0",
        "notes": "Fair local AIGC edit diagnostic; unmasked region is copied exactly from watermarked source.",
    }
    started = time.time()
    try:
        watermarked = load_rgb(watermarked_path)
        diffusion = load_rgb(diffusion_path)
        mask = load_mask(edit_mask_src, mask_threshold)
        if watermarked.shape != diffusion.shape:
            raise ValueError(f"size mismatch: watermarked={watermarked.shape}, diffusion={diffusion.shape}")
        if int(mask.sum()) == 0:
            raise ValueError("empty edit mask")
        composite = watermarked.copy()
        composite[mask] = diffusion[mask]
        outside = ~mask
        inside = mask
        diff_out = np.abs(composite.astype(np.int16) - watermarked.astype(np.int16))
        diff_in = np.abs(composite.astype(np.int16) - watermarked.astype(np.int16))
        outside_mean = float(np.mean(diff_out[outside])) if int(outside.sum()) else 0.0
        outside_max = int(np.max(diff_out[outside])) if int(outside.sum()) else 0
        inside_mean = float(np.mean(diff_in[inside])) if int(inside.sum()) else 0.0
        mask_ratio = float(np.mean(mask))
        save_rgb(out_dir / "original.png", load_rgb(original_path))
        save_rgb(out_dir / "watermarked.png", watermarked)
        save_rgb(composite_path, composite)
        save_mask(out_dir / "gt_mask.png", mask)
        save_mask(out_dir / "edit_mask.png", mask)
        save_mask(edit_mask_out, mask)
        copy_if_exists(src_dir / "payload.json", out_dir / "payload.json")
        meta = {
            "sample_id": sample_id,
            "source_manifest_row": row,
            "localized_composite_rule": "tampered_local = mask * diffusion_generated + (1-mask) * watermarked_image",
            "verification_original_access": False,
            "gt_mask_usage": "evaluation_only",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        out_row.update(
            {
                "mask_area_ratio": mask_ratio,
                "outside_region_preserved_check": outside_max <= 1 and outside_mean <= 1e-6,
                "outside_mean_abs_diff": outside_mean,
                "outside_max_abs_diff": outside_max,
                "inside_region_changed_check": inside_mean > 1.0,
                "inside_mean_abs_diff": inside_mean,
                "image_size_match": True,
                "generation_status": "ok",
                "runtime_seconds": time.time() - started,
                "tamper_changed_pixel_ratio": float(np.mean(np.mean(diff_in, axis=2)[inside] > 3.0)),
            }
        )
    except Exception as exc:
        out_row["error_message"] = f"{type(exc).__name__}: {exc}"
    return out_row


def save_contact_sheet(rows: list[dict[str, Any]], out_path: Path, max_items: int) -> None:
    ok = [row for row in rows if row.get("generation_status") == "ok"]
    if not ok:
        return
    if len(ok) <= max_items:
        selected = ok
    else:
        selected = [ok[round(i * (len(ok) - 1) / (max_items - 1))] for i in range(max_items)]
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


def run_eval(project_root: Path, output_dir: Path, eval_dir: Path, manifest_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_dir / "stage10a_lc_eval.log"
    command = [
        args.python_executable,
        str(SCRIPT_DIR / "run_stage10a_evaluate_real_aigc_attacks.py"),
        "--project_root",
        str(project_root),
        "--attack_dir",
        str(output_dir),
        "--manifest",
        str(manifest_path),
        "--output_dir",
        str(eval_dir),
        "--max_samples",
        str(args.max_samples),
        "--python_executable",
        args.python_executable,
        "--contact_sheet_samples",
        str(args.contact_sheet_samples),
        "--overwrite",
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(project_root), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    return {"command": command, "returncode": proc.returncode, "runtime_seconds": time.time() - started, "log_path": str(log_path)}


def copy_eval_aliases(eval_dir: Path) -> None:
    aliases = {
        "stage10a_eval_summary.json": "stage10a_lc_eval_summary.json",
        "stage10a_eval_by_attack_type.csv": "stage10a_lc_eval_by_attack_type.csv",
        "stage10a_strategy_comparison.csv": "stage10a_lc_strategy_comparison.csv",
        "stage10a_failure_cases.json": "stage10a_lc_failure_cases.json",
        "stage10a_payload_auth_failure_analysis.json": "stage10a_lc_payload_auth_failure_analysis.json",
        "stage10a_unknown_analysis.json": "stage10a_lc_unknown_analysis.json",
        "stage10a_examples_contact_sheet.png": "stage10a_lc_examples_contact_sheet.png",
        "stage10a_eval_overview.md": "stage10a_lc_eval_overview.md",
    }
    for src_name, dst_name in aliases.items():
        src = eval_dir / src_name
        if src.exists():
            shutil.copyfile(src, eval_dir / dst_name)
    patch_lc_eval_summary_to_f3(eval_dir)


def patch_lc_eval_summary_to_f3(eval_dir: Path) -> None:
    summary_path = eval_dir / "stage10a_lc_eval_summary.json"
    if not summary_path.exists():
        return
    summary = load_json(summary_path)
    f3 = strategy_summary(eval_dir, "F3_merge_nearby_fragments")
    if not f3:
        return
    original_report_quality = dict(summary.get("report_quality", {}))
    summary["stage8f_best_by_selection_report_quality"] = original_report_quality
    summary["stage8f_main_strategy"] = "F3_merge_nearby_fragments"
    summary["report_quality"] = {
        "unknown_before_stage8e": original_report_quality.get("unknown_before_stage8e"),
        "unknown_after_stage8e": original_report_quality.get("unknown_after_stage8e"),
        "f3_unknown_rate": safe_float(f3.get("unknown_rate_after")),
        "f3_reports": int(safe_float(f3.get("total_filtered_regions"))),
        "f3_reports_per_image": safe_float(f3.get("mean_reports_per_image")),
        "f3_gt_coverage_retained": safe_float(f3.get("mean_gt_tamper_area_coverage")),
        "f3_predicted_area_retained": safe_float(f3.get("mean_pred_area_retained")),
        "f3_no_report_images": int(safe_float(f3.get("no_report_images"))),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def setting_row_from_eval(setting: str, eval_dir: Path, pred_area: float | None, interpretation: str) -> dict[str, Any]:
    row = eval_metrics_row(setting, eval_dir, pred_area_from_eval_dir(eval_dir, argparse.Namespace(mask_threshold=127)))
    summary = load_json(eval_dir / "stage10a_lc_eval_summary.json") or load_json(eval_dir / "stage10a_eval_summary.json")
    f3 = strategy_summary(eval_dir, "F3_merge_nearby_fragments")
    row["capsule_recovery"] = summary.get("stage8c_payload_recovery", {}).get("mean_capsule_recovery_accuracy")
    if f3:
        row["f3_unknown"] = safe_float(f3.get("unknown_rate_after"))
        row["reports_per_image"] = safe_float(f3.get("mean_reports_per_image"))
        row["gt_coverage"] = safe_float(f3.get("mean_gt_tamper_area_coverage"))
        row["predicted_area_retained"] = safe_float(f3.get("mean_pred_area_retained"))
    else:
        row["predicted_area_retained"] = summary.get("report_quality", {}).get("f3_predicted_area_retained")
    if pred_area is not None:
        row["predicted_mask_area_ratio"] = pred_area
    row["interpretation"] = interpretation
    return row


def stage9_row(stage9_dir: Path) -> dict[str, Any]:
    row = stage9_metrics(stage9_dir)
    row["capsule_recovery"] = load_json(stage9_dir / "stage9_1000_key_metrics.json").get("capsule_recovery", 0.99965)
    row["predicted_area_retained"] = 1.0
    row["interpretation"] = "synthetic local tamper full evaluation"
    return row


def four_setting_comparison(
    stage9_dir: Path,
    unconstrained_eval_dir: Path,
    audit_dir: Path,
    lc_eval_dir: Path,
    out_path: Path,
) -> list[dict[str, Any]]:
    audit_summary = load_json(audit_dir / "stage10a_attack_locality_audit_summary.json")
    diag_summary = audit_summary.get("localized_composite_40", {})
    unconstrained_pred = audit_summary.get("predicted_mask", {}).get("mean_predicted_mask_area_ratio", 1.0)
    rows = [
        stage9_row(stage9_dir),
        setting_row_from_eval(
            "Stage10A_unconstrained_diffusion_200",
            unconstrained_eval_dir,
            unconstrained_pred,
            "global diffusion re-rendering / watermark laundering stress test",
        ),
        {
            "setting": "Stage10A_localized_composite_diagnostic_40",
            "iou": diag_summary.get("iou"),
            "dice": diag_summary.get("dice"),
            "precision": diag_summary.get("precision"),
            "recall": diag_summary.get("recall"),
            "bit_accuracy": diag_summary.get("bit_accuracy"),
            "payload_recovery": diag_summary.get("payload_recovery"),
            "capsule_recovery": load_json(audit_dir / "localized_composite_40_eval" / "stage10a_eval_summary.json")
            .get("stage8c_payload_recovery", {})
            .get("mean_capsule_recovery_accuracy"),
            "auth_success": diag_summary.get("auth_success"),
            "f3_unknown": diag_summary.get("f3_unknown"),
            "reports_per_image": diag_summary.get("reports_per_image"),
            "gt_coverage": diag_summary.get("gt_coverage"),
            "predicted_area_retained": 1.0,
            "predicted_mask_area_ratio": diag_summary.get("predicted_mask_area_ratio"),
            "interpretation": "40-sample diagnostic preserving unmasked watermarked pixels",
        },
        setting_row_from_eval(
            "Stage10A_localized_composite_full_200",
            lc_eval_dir,
            None,
            "fair local AIGC editing evaluation preserving unmasked watermarked pixels",
        ),
    ]
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


def strategy_summary(eval_dir: Path, strategy: str) -> dict[str, Any]:
    path = eval_dir / "stage10a_lc_strategy_comparison.csv"
    if not path.exists():
        path = eval_dir / "stage10a_strategy_comparison.csv"
    if not path.exists():
        return {}
    for row in read_csv(path):
        if row.get("strategy") == strategy:
            return row
    return {}


def write_overview_and_note(
    project_root: Path,
    output_dir: Path,
    eval_dir: Path,
    results_summary_dir: Path,
    construction_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    eval_record: dict[str, Any],
) -> None:
    loc = eval_summary.get("localization", {})
    report = eval_summary.get("report_quality", {})
    payload = eval_summary.get("stage8c_payload_recovery", {})
    f4_top3 = strategy_summary(eval_dir, "F4_main_region_top3")
    f4_top5 = strategy_summary(eval_dir, "F4_main_region_top5")
    lines = [
        "# Stage 10A Localized-Composite Full 200 Note",
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
        "## Construction",
        "",
        f"- completed cases: {construction_summary.get('constructed_cases')} / {construction_summary.get('total_cases')}",
        f"- failed construction: {construction_summary.get('failed_cases')}",
        f"- outside-mask preserved pass count: {construction_summary.get('outside_region_preserved_pass_count')}",
        f"- inside-mask changed pass count: {construction_summary.get('inside_region_changed_pass_count')}",
        f"- cases by attack type: {construction_summary.get('cases_by_attack_type')}",
        "",
        "## Evaluation",
        "",
        f"- completed: {eval_summary.get('completed')}",
        f"- failed_stage: {eval_summary.get('failed_stage')}",
        f"- eval runtime seconds: {eval_summary.get('runtime_total_seconds')}",
        f"- IoU: {loc.get('iou')}",
        f"- Dice: {loc.get('dice')}",
        f"- Precision: {loc.get('precision')}",
        f"- Recall: {loc.get('recall')}",
        f"- bit accuracy: {payload.get('mean_bit_accuracy')}",
        f"- payload recovery: {payload.get('mean_payload_recovery_accuracy')}",
        f"- capsule recovery: {payload.get('mean_capsule_recovery_accuracy')}",
        f"- auth success: {payload.get('mean_auth_success')}",
        f"- F3 unknown: {report.get('f3_unknown_rate')}",
        f"- F3 reports/image: {report.get('f3_reports_per_image')}",
        f"- GT coverage retained: {report.get('f3_gt_coverage_retained')}",
        f"- predicted area retained: {report.get('f3_predicted_area_retained')}",
        f"- no-report images: {report.get('f3_no_report_images')}",
        f"- F4_top3 mean_reports_per_image: {f4_top3.get('mean_reports_per_image')}",
        f"- F4_top3 GT coverage: {f4_top3.get('mean_gt_tamper_area_coverage')}",
        f"- F4_top5 mean_reports_per_image: {f4_top5.get('mean_reports_per_image')}",
        f"- F4_top5 GT coverage: {f4_top5.get('mean_gt_tamper_area_coverage')}",
        "",
        "## Interpretation",
        "",
        "Localized-composite full 200 preserves the unmasked watermarked region and should be treated as the fair local AIGC editing evaluation. The original Stage10A unconstrained diffusion result should be retained as a global diffusion re-rendering / watermark laundering stress test.",
    ]
    note = project_root / "STAGE10A_LOCALIZED_COMPOSITE_FULL200_NOTE.md"
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_dir.joinpath("stage10a_localized_composite_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    results_summary_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(note, results_summary_dir / "stage10a_lc_full200_note.md")
    for src, dst in [
        (output_dir / "stage10a_localized_composite_generation_summary.json", results_summary_dir / "stage10a_lc_generation_summary.json"),
        (eval_dir / "stage10a_lc_eval_summary.json", results_summary_dir / "stage10a_lc_eval_summary.json"),
        (eval_dir / "stage10a_lc_eval_by_attack_type.csv", results_summary_dir / "stage10a_lc_eval_by_attack_type.csv"),
        (eval_dir / "stage10a_lc_strategy_comparison.csv", results_summary_dir / "stage10a_lc_strategy_comparison.csv"),
        (eval_dir / "stage10a_lc_failure_cases.json", results_summary_dir / "stage10a_lc_failure_cases.json"),
        (eval_dir / "stage10a_lc_payload_auth_failure_analysis.json", results_summary_dir / "stage10a_lc_payload_auth_failure_analysis.json"),
        (eval_dir / "stage10a_lc_unknown_analysis.json", results_summary_dir / "stage10a_lc_unknown_analysis.json"),
        (project_root / "dfg_locguard" / "outputs" / "stage10a_localized_composite_full_200_eval" / "stage10a_four_setting_comparison.csv", results_summary_dir / "stage10a_four_setting_comparison.csv"),
    ]:
        if src.exists():
            shutil.copyfile(src, dst)
    summary_payload = {
        "construction_summary": construction_summary,
        "evaluation_summary": eval_summary,
        "eval_record": eval_record,
        "four_setting_comparison": comparison_rows,
    }
    (results_summary_dir / "stage10a_lc_full200_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    source_attack_dir = resolve(project_root, args.source_attack_dir)
    source_eval_dir = resolve(project_root, args.source_eval_dir)
    audit_dir = resolve(project_root, args.audit_dir)
    stage9_dir = resolve(project_root, args.stage9_dir)
    output_dir = resolve(project_root, args.output_dir)
    eval_dir = resolve(project_root, args.eval_output_dir)
    results_summary_dir = resolve(project_root, args.results_summary_dir)

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "localized_attacks").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    if args.overwrite and eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = source_attack_dir / "stage10a_attack_manifest.csv"
    rows = [row for row in read_csv(source_manifest) if row.get("generation_status") == "ok"]
    rows = rows[: args.max_samples] if args.max_samples else rows
    constructed: list[dict[str, Any]] = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        constructed.append(build_localized_row(row, output_dir, args.mask_threshold))
        if index % 50 == 0 or index == len(rows):
            failed = sum(1 for item in constructed if item.get("generation_status") != "ok")
            print(f"Stage10A localized composite {index}/{len(rows)} failed={failed}", flush=True)

    manifest_path = output_dir / "stage10a_localized_composite_manifest.csv"
    write_csv(manifest_path, constructed, MANIFEST_FIELDS)
    save_contact_sheet(constructed, output_dir / "stage10a_localized_composite_contact_sheet.png", args.contact_sheet_samples)
    ok_rows = [row for row in constructed if row.get("generation_status") == "ok"]
    failed_rows = [row for row in constructed if row.get("generation_status") != "ok"]
    construction_summary = {
        "stage": "stage10a_localized_composite_full_200_construction",
        "total_cases": len(constructed),
        "constructed_cases": len(ok_rows),
        "failed_cases": len(failed_rows),
        "cases_by_attack_type": dict(Counter(row.get("attack_type") for row in ok_rows)),
        "construction_runtime_seconds": time.time() - started,
        "mean_mask_area_ratio": mean([row.get("mask_area_ratio") for row in ok_rows]),
        "outside_region_preserved_pass_count": sum(1 for row in ok_rows if str(row.get("outside_region_preserved_check")).lower() == "true" or row.get("outside_region_preserved_check") is True),
        "inside_region_changed_pass_count": sum(1 for row in ok_rows if str(row.get("inside_region_changed_check")).lower() == "true" or row.get("inside_region_changed_check") is True),
        "failed_samples": failed_rows,
        "rule": "tampered_local = mask * diffusion_generated + (1-mask) * watermarked_image",
        "constraints": {
            "no_training": True,
            "no_vlm": True,
            "no_128bit_payload": True,
            "watermark_embedding_modified": False,
            "robust_payload_bits": 64,
        },
    }
    (output_dir / "stage10a_localized_composite_generation_summary.json").write_text(
        json.dumps(construction_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if len(ok_rows) != len(constructed):
        raise RuntimeError(f"Localized composite construction failed for {len(failed_rows)} samples")

    eval_record: dict[str, Any] = {"skipped": args.skip_eval}
    if not args.skip_eval:
        eval_record = run_eval(project_root, output_dir, eval_dir, manifest_path, args)
        if eval_record["returncode"] != 0:
            raise RuntimeError(f"Localized composite evaluation failed: {eval_record}")
    copy_eval_aliases(eval_dir)

    eval_summary = load_json(eval_dir / "stage10a_lc_eval_summary.json")
    comparison_rows = four_setting_comparison(
        stage9_dir,
        source_eval_dir,
        audit_dir,
        eval_dir,
        eval_dir / "stage10a_four_setting_comparison.csv",
    )
    write_overview_and_note(project_root, output_dir, eval_dir, results_summary_dir, construction_summary, eval_summary, comparison_rows, eval_record)
    print(json.dumps({"construction": construction_summary, "evaluation": eval_summary, "eval_record": eval_record}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
