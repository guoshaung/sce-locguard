from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_editguard_outputs_stage1 as stage1
import run_stage7c_non_oracle_semantic_template as s7c
import run_stage8c_robust_payload_semantic_capsule_audit as s8c


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
NEAREST = getattr(Image, "Resampling", Image).NEAREST

ATTACK_TYPES = ["object_removal", "inpainting", "local_replacement", "local_style_edit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10A evaluation for real AIGC editing attacks.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--attack_dir", default="dfg_locguard/outputs/stage10a_real_aigc_attack_subset/full_200")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage10a_real_aigc_attack_eval")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--recovery_threshold", type=float, default=0.2)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def mean(values: list[Any]) -> float:
    nums = [safe_float(v) for v in values if v not in ("", None)]
    return float(sum(nums) / len(nums)) if nums else 0.0


def copy_image(src: Path, dst: Path, mode: str = "RGB") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert(mode).resize((512, 512), BICUBIC if mode == "RGB" else NEAREST).save(dst)


def recovered_bits_from_tensor(remesg: Any) -> list[int]:
    return s8c.recovered_bits_from_tensor(remesg)


def build_stage8c_real_row(
    model: Any,
    manifest_row: dict[str, str],
    sample_dir: Path,
    output_sample_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = json.loads((sample_dir / "payload.json").read_text(encoding="utf-8"))
    tampered = np.asarray(Image.open(sample_dir / "tampered.png").convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)
    gt_mask = np.asarray(Image.open(sample_dir / "gt_mask.png").convert("L").resize((512, 512), NEAREST), dtype=np.uint8) > args.mask_threshold
    watermarked = np.asarray(Image.open(sample_dir / "watermarked.png").convert("RGB").resize((512, 512), BICUBIC), dtype=np.uint8)

    tampered_data = stage1.image_to_editguard_batch(tampered)
    tampered_data["MES"] = None
    model.feed_data(tampered_data)
    pred_raw, remesg = model.image_recovery(args.recovery_threshold)
    pred_mask = np.asarray(pred_raw) > 0
    Image.fromarray(pred_mask.astype(np.uint8) * 255).save(output_sample_dir / "editguard_pred_mask.png")

    recovered_bits = recovered_bits_from_tensor(remesg)
    recovery = s8c.evaluate_recovery(payload, recovered_bits)
    pix = s7c.pixel_metrics(gt_mask, pred_mask)
    comp = s7c.component_metrics(gt_mask, pred_mask, 100, 25)
    image_id = manifest_row["sample_id"]
    row: dict[str, Any] = {
        "payload_variant": "P2_id_plus_compact_capsule_64",
        "image_id": image_id,
        "attack_type": manifest_row.get("attack_type", ""),
        "status": "ok",
        "error_message": "",
        "payload_mode": "direct",
        "payload_bits": len(payload["bits"]),
        "target_payload_bits": 64,
        "supported_by_current_model": True,
        "custom_payload_injected": True,
        "payload_spec_name": "P2_id_plus_compact_capsule_64",
        "payload_fields": json.dumps(payload["fields"], sort_keys=True),
        "copyright_id": payload["copyright_id"],
        "semantic_capsule_json": json.dumps(payload["semantic_capsule"], sort_keys=True),
        "checksum_bits": payload["checksum_bits"],
        "auth_tag_bits": payload["auth_tag_bits"],
        "original_path": str((output_sample_dir / "original.png").resolve()),
        "gt_mask_path": str((output_sample_dir / "gt_mask.png").resolve()),
        "watermarked_path": str((output_sample_dir / "watermarked.png").resolve()),
        "tampered_path": str((output_sample_dir / "tampered.png").resolve()),
        "pred_mask_path": str((output_sample_dir / "editguard_pred_mask.png").resolve()),
        "copyright_eval_available": True,
        "psnr": s7c.psnr(watermarked, tampered),
        "ssim": s7c.ssim_rgb(watermarked, tampered),
    }
    row.update(recovery)
    row.update(pix)
    row.update(comp)
    return row


def run_stage(name: str, command: list[str], project_root: Path, log_dir: Path) -> dict[str, Any]:
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=str(project_root), stdout=out, stderr=err, text=True, check=False)
    return {
        "stage": name,
        "returncode": proc.returncode,
        "runtime_seconds": time.time() - started,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command": command,
    }


def summarize_by_attack_type(
    manifest_rows: list[dict[str, str]],
    stage8c_rows: list[dict[str, Any]],
    stage8d_rows: list[dict[str, str]],
    stage8f_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    attack_of = {row["sample_id"]: row["attack_type"] for row in manifest_rows}
    s8c_by_id = {row["image_id"]: row for row in stage8c_rows if row.get("payload_variant") == "P2_id_plus_compact_capsule_64"}
    s8d_by_id = {row["image_id"]: row for row in stage8d_rows if row.get("status") == "ok"}
    f3 = [row for row in stage8f_rows if row.get("strategy") == "F3_merge_nearby_fragments"]
    f3_by_id = {row["image_id"]: row for row in f3}
    rows: list[dict[str, Any]] = []
    for attack_type in ATTACK_TYPES:
        ids = [sample_id for sample_id, at in attack_of.items() if at == attack_type]
        rows.append(
            {
                "attack_type": attack_type,
                "sample_count": len(ids),
                "iou": mean([s8d_by_id.get(i, {}).get("iou") for i in ids]),
                "dice": mean([s8d_by_id.get(i, {}).get("dice") for i in ids]),
                "payload_recovery": mean([s8c_by_id.get(i, {}).get("payload_recovery_accuracy") for i in ids]),
                "auth_success": mean([s8c_by_id.get(i, {}).get("auth_check_success") for i in ids]),
                "bit_accuracy": mean([s8c_by_id.get(i, {}).get("bit_accuracy") for i in ids]),
                "capsule_recovery": mean([s8c_by_id.get(i, {}).get("semantic_capsule_recovery_accuracy") for i in ids]),
                "f3_unknown_rate": mean([f3_by_id.get(i, {}).get("unknown_rate_after_filter") for i in ids]),
                "f3_reports_per_image": mean([f3_by_id.get(i, {}).get("num_filtered_regions") for i in ids]),
                "gt_coverage_retained": mean([f3_by_id.get(i, {}).get("gt_tamper_area_coverage") for i in ids]),
                "predicted_area_retained": mean([f3_by_id.get(i, {}).get("pred_area_retained") for i in ids]),
                "no_report_images": sum(1 for i in ids if int(float(f3_by_id.get(i, {}).get("num_no_report_image", 0) or 0)) > 0),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    attack_dir = resolve(project_root, args.attack_dir)
    manifest_path = resolve(project_root, args.manifest) if args.manifest else attack_dir / "stage10a_attack_manifest.csv"
    output_dir = resolve(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stage1_like = output_dir / "stage10a_stage1_like_inputs"
    stage8c_dir = output_dir / "stage8c_real_payload_recovery"
    stage8d_dir = output_dir / "stage8d"
    stage8e_dir = output_dir / "stage8e"
    stage8f_dir = output_dir / "stage8f"
    for path in [stage1_like, stage8c_dir]:
        if args.overwrite and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows_all = read_csv(manifest_path)
    manifest_rows = [row for row in manifest_rows_all if row.get("generation_status") == "ok"]
    if args.max_samples:
        manifest_rows = manifest_rows[: args.max_samples]
    if not manifest_rows:
        raise FileNotFoundError(f"No generated attack rows found in {manifest_path}")

    model = stage1.load_editguard_model(project_root, project_root / args.opt, project_root / args.ckpt)
    rows8c: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    start_total = time.time()
    for index, row in enumerate(manifest_rows, start=1):
        sample_id = row["sample_id"]
        src_dir = Path(row["generated_tampered_image_path"]).resolve().parent
        out_sample = stage1_like / sample_id
        out_sample.mkdir(parents=True, exist_ok=True)
        try:
            copy_image(src_dir / "original.png", out_sample / "original.png", "RGB")
            copy_image(src_dir / "watermarked.png", out_sample / "watermarked.png", "RGB")
            copy_image(src_dir / "tampered.png", out_sample / "tampered.png", "RGB")
            copy_image(src_dir / "gt_mask.png", out_sample / "gt_mask.png", "L")
            meta = {"sample_id": sample_id, "attack_type": row["attack_type"], "source_manifest_row": row}
            (out_sample / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            rows8c.append(build_stage8c_real_row(model, row, src_dir, out_sample, args))
        except Exception as exc:
            failed.append({"sample_id": sample_id, "attack_type": row.get("attack_type"), "failure_stage": "stage10a_recovery", "error_message": str(exc)})
        if index % 20 == 0 or index == len(manifest_rows):
            print(f"Stage10A recovery {index}/{len(manifest_rows)} failed={len(failed)}", flush=True)

    s8c_fields = list(dict.fromkeys(s8c.CSV_FIELDS + ["attack_type"]))
    write_csv(stage8c_dir / "per_sample_stage8c_metrics.csv", rows8c, s8c_fields)
    (stage8c_dir / "stage8c_failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    s8c_summary = {
        "stage": "stage10a_real_attack_payload_recovery",
        "total_samples": len(manifest_rows),
        "evaluated_samples": len(rows8c),
        "failed_cases": len(failed),
        "payload_variant": "P2_id_plus_compact_capsule_64",
        "mean_bit_accuracy": mean([row.get("bit_accuracy") for row in rows8c]),
        "mean_payload_recovery_accuracy": mean([row.get("payload_recovery_accuracy") for row in rows8c]),
        "mean_capsule_recovery_accuracy": mean([row.get("semantic_capsule_recovery_accuracy") for row in rows8c]),
        "mean_auth_success": mean([row.get("auth_check_success") for row in rows8c]),
    }
    (stage8c_dir / "stage8c_summary.json").write_text(json.dumps(s8c_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    commands = [
        (
            "stage8d_dual_branch_report",
            [
                args.python_executable,
                str(SCRIPT_DIR / "run_stage8d_end_to_end_dual_branch_report.py"),
                "--project_root",
                str(project_root),
                "--stage1_dir",
                str(stage1_like),
                "--stage8c_dir",
                str(stage8c_dir),
                "--output_dir",
                str(stage8d_dir),
                "--max_samples",
                str(len(manifest_rows)),
                "--mask_threshold",
                str(args.mask_threshold),
                "--contact_sheet_samples",
                str(args.contact_sheet_samples),
            ],
        ),
        (
            "stage8e_report_quality_audit",
            [
                args.python_executable,
                str(SCRIPT_DIR / "run_stage8e_report_quality_audit.py"),
                "--project_root",
                str(project_root),
                "--stage8d_dir",
                str(stage8d_dir),
                "--stage1_dir",
                str(stage1_like),
                "--output_dir",
                str(stage8e_dir),
                "--max_samples",
                str(len(manifest_rows)),
                "--contact_sheet_samples",
                str(args.contact_sheet_samples),
            ],
        ),
        (
            "stage8f_report_region_aggregation",
            [
                args.python_executable,
                str(SCRIPT_DIR / "run_stage8f_report_region_aggregation.py"),
                "--project_root",
                str(project_root),
                "--stage8e_dir",
                str(stage8e_dir),
                "--stage1_dir",
                str(stage1_like),
                "--output_dir",
                str(stage8f_dir),
                "--max_samples",
                str(len(manifest_rows)),
                "--mask_threshold",
                str(args.mask_threshold),
                "--contact_sheet_samples",
                str(args.contact_sheet_samples),
            ],
        ),
    ]
    if args.overwrite:
        commands = [(name, command + ["--overwrite"]) for name, command in commands]

    run_records = []
    failed_stage = None
    for name, command in commands:
        record = run_stage(name, command, project_root, log_dir)
        run_records.append(record)
        if record["returncode"] != 0:
            failed_stage = name
            break

    stage8d_summary = json.loads((stage8d_dir / "stage8d_summary.json").read_text(encoding="utf-8")) if (stage8d_dir / "stage8d_summary.json").exists() else {}
    stage8e_summary = json.loads((stage8e_dir / "stage8e_quality_summary.json").read_text(encoding="utf-8")) if (stage8e_dir / "stage8e_quality_summary.json").exists() else {}
    stage8f_summary = json.loads((stage8f_dir / "stage8f_summary.json").read_text(encoding="utf-8")) if (stage8f_dir / "stage8f_summary.json").exists() else {}
    best = stage8f_summary.get("best_strategy_payload", {})
    stage8d_rows = read_csv(stage8d_dir / "per_sample_stage8d_summary.csv") if (stage8d_dir / "per_sample_stage8d_summary.csv").exists() else []
    stage8f_rows = read_csv(stage8f_dir / "per_sample_stage8f_summary.csv") if (stage8f_dir / "per_sample_stage8f_summary.csv").exists() else []
    by_attack = summarize_by_attack_type(manifest_rows, rows8c, stage8d_rows, stage8f_rows)
    write_csv(
        output_dir / "stage10a_eval_by_attack_type.csv",
        by_attack,
        [
            "attack_type",
            "sample_count",
            "iou",
            "dice",
            "payload_recovery",
            "auth_success",
            "bit_accuracy",
            "capsule_recovery",
            "f3_unknown_rate",
            "f3_reports_per_image",
            "gt_coverage_retained",
            "predicted_area_retained",
            "no_report_images",
        ],
    )
    if (stage8f_dir / "stage8f_strategy_comparison.csv").exists():
        shutil.copyfile(stage8f_dir / "stage8f_strategy_comparison.csv", output_dir / "stage10a_strategy_comparison.csv")
    if (stage8e_dir / "stage8e_unknown_analysis.json").exists():
        shutil.copyfile(stage8e_dir / "stage8e_unknown_analysis.json", output_dir / "stage10a_unknown_analysis.json")
    if (stage8f_dir / "stage8f_contact_sheet.png").exists():
        shutil.copyfile(stage8f_dir / "stage8f_contact_sheet.png", output_dir / "stage10a_examples_contact_sheet.png")

    payload_failed = [row["image_id"] for row in rows8c if safe_float(row.get("payload_recovery_accuracy")) < 0.999]
    auth_failed = [row["image_id"] for row in rows8c if safe_float(row.get("auth_check_success")) < 0.999]
    payload_analysis = {
        "payload_failed_count": len(payload_failed),
        "auth_failed_count": len(auth_failed),
        "payload_failed_sample_ids": payload_failed,
        "auth_failed_sample_ids": auth_failed,
        "by_attack_type": by_attack,
    }
    (output_dir / "stage10a_payload_auth_failure_analysis.json").write_text(json.dumps(payload_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "stage10a_failure_cases.json").write_text(json.dumps({"recovery_failed": failed, "pipeline_failed_stage": failed_stage}, ensure_ascii=False, indent=2), encoding="utf-8")
    total_runtime = time.time() - start_total
    summary = {
        "stage": "stage10a_real_aigc_attack_eval",
        "completed": failed_stage is None,
        "failed_stage": failed_stage,
        "total_samples": len(manifest_rows),
        "samples_by_attack_type": {attack_type: sum(1 for row in manifest_rows if row["attack_type"] == attack_type) for attack_type in ATTACK_TYPES},
        "failed_generation": len(manifest_rows_all) - len(manifest_rows),
        "failed_evaluation": len(failed),
        "runtime_total_seconds": total_runtime,
        "runtime_per_sample_seconds": total_runtime / len(manifest_rows) if manifest_rows else 0.0,
        "stage8c_payload_recovery": s8c_summary,
        "localization": {
            "iou": stage8d_summary.get("mean_iou"),
            "dice": stage8d_summary.get("mean_dice"),
            "precision": stage8d_summary.get("mean_precision"),
            "recall": stage8d_summary.get("mean_recall"),
            "mae": stage8d_summary.get("mean_mae"),
        },
        "report_quality": {
            "unknown_before_stage8e": stage8e_summary.get("unknown_change_type_rate_original"),
            "unknown_after_stage8e": stage8e_summary.get("unknown_change_type_rate_improved"),
            "f3_unknown_rate": best.get("unknown_rate_after"),
            "f3_reports_per_image": best.get("mean_reports_per_image"),
            "f3_gt_coverage_retained": best.get("mean_gt_tamper_area_coverage"),
            "f3_predicted_area_retained": best.get("mean_pred_area_retained"),
            "f3_no_report_images": best.get("no_report_images"),
        },
        "by_attack_type": by_attack,
        "run_records": run_records,
        "constraints": {
            "no_training": True,
            "no_vlm": True,
            "no_128bit_payload": True,
            "watermark_embedding_modified": False,
            "verification_original_access": False,
            "gt_mask_usage": ["evaluation_only"],
        },
    }
    (output_dir / "stage10a_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    overview = [
        "# Stage 10A Real AIGC Attack Evaluation",
        "",
        f"- completed: {summary['completed']}",
        f"- total_samples: {summary['total_samples']}",
        f"- IoU: {summary['localization'].get('iou')}",
        f"- Dice: {summary['localization'].get('dice')}",
        f"- F3 unknown rate: {summary['report_quality'].get('f3_unknown_rate')}",
        f"- F3 reports/image: {summary['report_quality'].get('f3_reports_per_image')}",
        "",
        "Verification uses tampered image, predicted mask, and recovered robust payload. It does not access original images.",
    ]
    (output_dir / "stage10a_eval_overview.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
