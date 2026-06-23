#!/usr/bin/env python3
"""Stage 10B-1 pure EditGuard-style baseline extraction.

This script is intentionally read-only with respect to previous experiment
outputs. It extracts numeric localization and robust-payload metrics from
existing Stage9C / Stage10A outputs, then prepares paper-ready comparison
tables. It does not train, run VLMs, expand payload size, modify watermark
embedding, or generate images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METRIC_FIELDS = [
    "iou",
    "dice",
    "precision",
    "recall",
    "mae",
    "bit_accuracy",
    "payload_recovery_accuracy",
    "capsule_recovery_accuracy",
    "auth_success",
]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: normalize_csv_value(row.get(k)) for k in fieldnames})


def normalize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def mean(values: Iterable[Any]) -> Optional[float]:
    nums = [v for v in (to_float(x) for x in values) if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den in (None, 0):
        return None
    return num / den


def get_nested(payload: Dict[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def git_commit_hash(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class SettingSpec:
    key: str
    display_name: str
    description: str
    output_dir: Path
    summary_candidates: List[str]
    strategy_candidates: List[str]
    interpretation: str


def first_existing(base: Path, candidates: List[str]) -> Optional[Path]:
    for rel in candidates:
        path = base / rel
        if path.exists():
            return path
    return None


def summarize_stage8d_per_sample(path: Path) -> Dict[str, Any]:
    rows = read_csv_rows(path)
    ok_rows = [r for r in rows if (r.get("status") or "").lower() in ("ok", "success", "")]
    failed_rows = [r for r in rows if r not in ok_rows]
    summary: Dict[str, Any] = {
        "per_sample_path": str(path) if path.exists() else None,
        "sample_count": len(rows),
        "evaluated_samples": len(ok_rows),
        "failed_samples": len(failed_rows),
        "failed_sample_ids": [r.get("image_id") for r in failed_rows if r.get("image_id")],
    }
    source_rows = ok_rows or rows
    for field in METRIC_FIELDS:
        summary[field] = mean(r.get(field) for r in source_rows)
    return summary


def strategy_row(strategy_rows: List[Dict[str, str]], strategy: str) -> Dict[str, Any]:
    for row in strategy_rows:
        if row.get("strategy") == strategy:
            return dict(row)
    return {}


def extract_setting(spec: SettingSpec) -> Dict[str, Any]:
    summary_path = first_existing(spec.output_dir, spec.summary_candidates)
    summary_json = read_json(summary_path, {}) if summary_path else {}
    per_sample_path = spec.output_dir / "stage8d" / "per_sample_stage8d_summary.csv"
    per_sample = summarize_stage8d_per_sample(per_sample_path)
    strategy_path = first_existing(spec.output_dir, spec.strategy_candidates)
    strategy_rows = read_csv_rows(strategy_path) if strategy_path else []
    strategies = {r.get("strategy", ""): r for r in strategy_rows if r.get("strategy")}

    total_samples = (
        per_sample.get("sample_count")
        or summary_json.get("total_samples")
        or summary_json.get("max_samples")
        or summary_json.get("stage1_generated")
    )
    localization = summary_json.get("localization") or {}
    payload = summary_json.get("stage8c_payload_recovery") or {}

    def metric_value(name: str) -> Any:
        return (
            per_sample.get(name)
            if per_sample.get(name) is not None
            else localization.get(name)
            if localization.get(name) is not None
            else summary_json.get(name)
        )

    baseline = {
        "setting": spec.key,
        "setting_display_name": spec.display_name,
        "sample_count": total_samples,
        "iou": metric_value("iou"),
        "dice": metric_value("dice"),
        "precision": metric_value("precision"),
        "recall": metric_value("recall"),
        "mae": metric_value("mae"),
        "bit_accuracy": metric_value("bit_accuracy")
        if metric_value("bit_accuracy") is not None
        else payload.get("mean_bit_accuracy"),
        "payload_or_copyright_recovery": metric_value("payload_recovery_accuracy")
        if metric_value("payload_recovery_accuracy") is not None
        else payload.get("mean_payload_recovery_accuracy"),
        "auth_success": metric_value("auth_success")
        if metric_value("auth_success") is not None
        else payload.get("mean_auth_success"),
        "failed_samples": per_sample.get("failed_samples")
        if per_sample.get("sample_count")
        else summary_json.get("failed_evaluation", summary_json.get("stage1_failed")),
        "predicted_mask_area_ratio": summary_json.get("predicted_mask_area_ratio"),
        "gt_coverage": metric_value("recall"),
        "source_per_sample": str(per_sample_path),
        "source_summary": str(summary_path) if summary_path else "",
        "notes": (
            "Pure EditGuard-style baseline: fragile localization plus robust "
            "64-bit payload/copyright recovery only; excludes semantic capsule "
            "interpretation and Stage8E/Stage8F reports."
        ),
    }

    f0 = strategy_row(strategy_rows, "F0_no_filter")
    f3 = strategy_row(strategy_rows, "F3_merge_nearby_fragments")
    f4_top3 = strategy_row(strategy_rows, "F4_main_region_top3")
    f4_top5 = strategy_row(strategy_rows, "F4_main_region_top5")

    report_quality = summary_json.get("report_quality") or {}
    if "stage8e_unknown_after" in summary_json:
        stage8e_unknown = summary_json.get("stage8e_unknown_after")
        stage8e_unknown_before = summary_json.get("stage8e_unknown_before")
    else:
        stage8e_unknown = report_quality.get("unknown_after_stage8e")
        stage8e_unknown_before = report_quality.get("unknown_before_stage8e")

    sce_common = {
        "sample_count": total_samples,
        "bit_accuracy": baseline["bit_accuracy"],
        "payload_recovery": baseline["payload_or_copyright_recovery"],
        "auth_success": baseline["auth_success"],
        "semantic_capsule_recovery": per_sample.get("capsule_recovery_accuracy")
        if per_sample.get("capsule_recovery_accuracy") is not None
        else payload.get("mean_capsule_recovery_accuracy")
        if payload
        else summary_json.get("capsule_recovery_accuracy"),
    }

    def sce_row(method_variant: str, strategy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        strategy = strategy or {}
        if strategy:
            iou = to_float(strategy.get("mean_aggregated_iou"))
            dice = to_float(strategy.get("mean_aggregated_dice"))
            gt_cov = to_float(strategy.get("mean_gt_tamper_area_coverage"))
            pred_retained = to_float(strategy.get("mean_pred_area_retained"))
            unknown_rate = to_float(strategy.get("unknown_rate_after"))
            reports_per_image = to_float(strategy.get("mean_reports_per_image"))
            no_report = to_float(strategy.get("no_report_images"))
        else:
            iou = baseline["iou"]
            dice = baseline["dice"]
            gt_cov = to_float(f0.get("mean_gt_tamper_area_coverage")) or baseline["recall"]
            pred_retained = to_float(f0.get("mean_pred_area_retained"))
            unknown_rate = to_float(stage8e_unknown)
            reports_per_image = safe_div(
                to_float(f0.get("total_filtered_regions")),
                to_float(f0.get("total_images")) or to_float(total_samples),
            )
            no_report = to_float(f0.get("no_report_images"))
        return {
            "setting": spec.key,
            "method_variant": method_variant,
            "sample_count": total_samples,
            "iou": iou,
            "dice": dice,
            "precision": baseline["precision"] if not strategy else "",
            "recall": baseline["recall"] if not strategy else "",
            "bit_accuracy": sce_common["bit_accuracy"],
            "payload_recovery": sce_common["payload_recovery"],
            "auth_success": sce_common["auth_success"],
            "semantic_capsule_recovery": sce_common["semantic_capsule_recovery"],
            "unknown_rate": unknown_rate,
            "reports_per_image": reports_per_image,
            "gt_coverage_retained": gt_cov,
            "predicted_area_retained": pred_retained,
            "no_report_images": no_report,
        }

    sce_rows = [
        {
            "setting": spec.key,
            "method_variant": "EditGuard-style baseline",
            "sample_count": total_samples,
            "iou": baseline["iou"],
            "dice": baseline["dice"],
            "precision": baseline["precision"],
            "recall": baseline["recall"],
            "bit_accuracy": baseline["bit_accuracy"],
            "payload_recovery": baseline["payload_or_copyright_recovery"],
            "auth_success": baseline["auth_success"],
            "semantic_capsule_recovery": "",
            "unknown_rate": "",
            "reports_per_image": "",
            "gt_coverage_retained": baseline["gt_coverage"],
            "predicted_area_retained": "",
            "no_report_images": "",
        },
        sce_row("SCE-LocGuard Stage8E"),
        sce_row("SCE-LocGuard Stage8F-F3", f3),
        sce_row("SCE-LocGuard Stage8F-F4_top3", f4_top3),
        sce_row("SCE-LocGuard Stage8F-F4_top5", f4_top5),
    ]

    return {
        "spec": spec,
        "summary_path": str(summary_path) if summary_path else None,
        "strategy_path": str(strategy_path) if strategy_path else None,
        "summary_json": summary_json,
        "baseline": baseline,
        "per_sample": per_sample,
        "strategies": strategies,
        "sce_rows": sce_rows,
        "stage8e_unknown_before": stage8e_unknown_before,
        "stage8e_unknown_after": stage8e_unknown,
    }


def make_capability_rows() -> List[Dict[str, Any]]:
    return [
        {
            "method": "EditGuard-style baseline",
            "tamper_localization": "yes",
            "copyright_provenance_recovery": "yes",
            "robust_payload": "64-bit copyright/payload only",
            "authentication": "available if payload checksum/auth is configured",
            "semantic_capsule": "no",
            "before_semantics_recovery": "no",
            "after_region_semantic_readout": "no",
            "change_type": "no",
            "severity": "no",
            "report_level_aggregation": "no",
            "compact_audit_level_report": "no",
            "failure_mode_attribution": "limited numeric failure only",
            "global_rerendering_stress_test_analysis": "localization/payload failure can be measured",
            "status": "extracted from existing outputs",
        },
        {
            "method": "SCE-LocGuard",
            "tamper_localization": "yes",
            "copyright_provenance_recovery": "yes",
            "robust_payload": "64-bit copyright ID plus compact semantic capsule/auth metadata",
            "authentication": "yes",
            "semantic_capsule": "yes",
            "before_semantics_recovery": "yes, from robust capsule",
            "after_region_semantic_readout": "yes, verification-side lightweight readout",
            "change_type": "yes, coarse rule-based",
            "severity": "yes, coarse rule-based",
            "report_level_aggregation": "yes, Stage8F",
            "compact_audit_level_report": "yes, F3/F4",
            "failure_mode_attribution": "yes, local edit vs global laundering interpretation",
            "global_rerendering_stress_test_analysis": "yes",
            "status": "implemented in current pipeline",
        },
        {
            "method": "OmniGuard",
            "tamper_localization": "yes",
            "copyright_provenance_recovery": "yes",
            "robust_payload": "reported by method; reproduction pending",
            "authentication": "unclear from lightweight inspection",
            "semantic_capsule": "no explicit SCE-style capsule confirmed",
            "before_semantics_recovery": "not confirmed",
            "after_region_semantic_readout": "not confirmed",
            "change_type": "not confirmed",
            "severity": "not confirmed",
            "report_level_aggregation": "not confirmed",
            "compact_audit_level_report": "not confirmed",
            "failure_mode_attribution": "partial if benchmark is reproduced",
            "global_rerendering_stress_test_analysis": "feasibility pending",
            "status": "feasibility pending",
        },
        {
            "method": "Dual-Guard",
            "tamper_localization": "yes, in paper",
            "copyright_provenance_recovery": "global provenance anchor",
            "robust_payload": "latent dual-channel anchors, not EditGuard-compatible payload",
            "authentication": "yes, in paper setting",
            "semantic_capsule": "no SCE-style capsule confirmed",
            "before_semantics_recovery": "not applicable",
            "after_region_semantic_readout": "not applicable",
            "change_type": "not primary output",
            "severity": "not primary output",
            "report_level_aggregation": "not primary output",
            "compact_audit_level_report": "not primary output",
            "failure_mode_attribution": "strong threat-model reference",
            "global_rerendering_stress_test_analysis": "yes, conceptual paper benchmark",
            "status": "conceptual related method unless code/checkpoint appears",
        },
    ]


def make_failure_rows(extractions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    templates = {
        "synthetic_local_tamper_1000": {
            "localization_behavior": "strong controlled local tamper localization",
            "payload_auth_behavior": "robust payload/auth mostly preserved",
            "semantic_report_behavior": "SCE semantic reports are available and compact after Stage8F",
            "failure_cause": "ordinary local tamper and mask fragmentation, not global laundering",
            "paper_interpretation": "primary controlled synthetic benchmark",
        },
        "localized_composited_aigc_local_edit_1000": {
            "localization_behavior": "very strong local edit localization when unmasked watermarked pixels are preserved",
            "payload_auth_behavior": "robust payload/auth mostly preserved",
            "semantic_report_behavior": "F3 reports remain compact but unknown rate rises under AIGC appearance changes",
            "failure_cause": "semantic readout ambiguity in small/tiny edited fragments, not watermark loss",
            "paper_interpretation": "fair local AIGC editing benchmark",
        },
        "unconstrained_diffusion_rerendering_200": {
            "localization_behavior": "predicted mask saturates / localization becomes non-informative",
            "payload_auth_behavior": "robust payload/auth collapses to random-level recovery",
            "semantic_report_behavior": "semantic report is not meaningful because watermark is globally laundered",
            "failure_cause": "global diffusion re-rendering / watermark laundering",
            "paper_interpretation": "stress test, not fair local edit benchmark",
        },
    }
    rows: List[Dict[str, Any]] = []
    for setting, text in templates.items():
        base = extractions[setting]["baseline"]
        rows.append(
            {
                "setting": setting,
                "iou": base.get("iou"),
                "dice": base.get("dice"),
                "bit_accuracy": base.get("bit_accuracy"),
                "payload_recovery": base.get("payload_or_copyright_recovery"),
                "auth_success": base.get("auth_success"),
                **text,
            }
        )
    return rows


def build_overview(
    output_dir: Path,
    baseline_rows: List[Dict[str, Any]],
    sce_rows: List[Dict[str, Any]],
    failure_rows: List[Dict[str, Any]],
) -> str:
    def fmt(x: Any) -> str:
        v = to_float(x)
        if v is None:
            return "NA"
        return f"{v:.4f}"

    lines = [
        "# Stage 10B-1 EditGuard-Style Baseline Extraction",
        "",
        "This stage extracts a pure EditGuard-style baseline from existing outputs only.",
        "It does not train, enable VLMs, expand the 64-bit payload, modify watermark embedding, or rerun large models.",
        "",
        "## Pure EditGuard-Style Baseline",
        "",
        "| Setting | Samples | IoU | Dice | Precision | Recall | Bit Acc | Payload/Auth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline_rows:
        lines.append(
            "| {setting} | {sample_count} | {iou} | {dice} | {precision} | {recall} | {bit} | {payload}/{auth} |".format(
                setting=row["setting"],
                sample_count=row.get("sample_count", ""),
                iou=fmt(row.get("iou")),
                dice=fmt(row.get("dice")),
                precision=fmt(row.get("precision")),
                recall=fmt(row.get("recall")),
                bit=fmt(row.get("bit_accuracy")),
                payload=fmt(row.get("payload_or_copyright_recovery")),
                auth=fmt(row.get("auth_success")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Synthetic local tamper is the controlled baseline benchmark.",
            "- Localized-composited AIGC local edit is the fair local AIGC editing benchmark because unmasked watermarked pixels are preserved.",
            "- Unconstrained diffusion re-rendering should be treated as a watermark-laundering stress test.",
            "",
            "## Output Files",
            "",
        ]
    )
    for name in [
        "stage10b_editguard_baseline_summary.json",
        "stage10b_editguard_baseline_by_setting.csv",
        "stage10b_method_capability_comparison.csv",
        "stage10b_sce_vs_editguard_numeric_comparison.csv",
        "stage10b_baseline_failure_mode_comparison.csv",
        "stage10b_editguard_baseline_overview.md",
    ]:
        lines.append(f"- `{output_dir / name}`")
    lines.extend(
        [
            "",
            "## Failure Mode Summary",
            "",
            "| Setting | Failure Cause | Paper Interpretation |",
            "|---|---|---|",
        ]
    )
    for row in failure_rows:
        lines.append(
            f"| {row['setting']} | {row['failure_cause']} | {row['paper_interpretation']} |"
        )
    return "\n".join(lines) + "\n"


def build_note(summary: Dict[str, Any]) -> str:
    rows = summary["editguard_style_baseline"]

    def lookup(setting: str) -> Dict[str, Any]:
        return next(r for r in rows if r["setting"] == setting)

    def fmt(x: Any) -> str:
        v = to_float(x)
        return "NA" if v is None else f"{v:.5f}"

    syn = lookup("synthetic_local_tamper_1000")
    lc = lookup("localized_composited_aigc_local_edit_1000")
    unc = lookup("unconstrained_diffusion_rerendering_200")
    return f"""# Stage 10B EditGuard Baseline Note

## Scope

- Pure EditGuard-style baseline extraction from existing Stage9C / Stage10A outputs.
- No training.
- No VLM.
- No 128-bit payload.
- No watermark embedding modification.
- Robust payload remains 64-bit.
- Stage8E/Stage8F semantic report metrics are excluded from the pure baseline.

## Pure EditGuard-Style Baseline Metrics

| Setting | Samples | IoU | Dice | Precision | Recall | Bit Acc | Payload Recovery | Auth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic local tamper 1000 | {syn.get('sample_count')} | {fmt(syn.get('iou'))} | {fmt(syn.get('dice'))} | {fmt(syn.get('precision'))} | {fmt(syn.get('recall'))} | {fmt(syn.get('bit_accuracy'))} | {fmt(syn.get('payload_or_copyright_recovery'))} | {fmt(syn.get('auth_success'))} |
| localized-composited AIGC local edit 1000 | {lc.get('sample_count')} | {fmt(lc.get('iou'))} | {fmt(lc.get('dice'))} | {fmt(lc.get('precision'))} | {fmt(lc.get('recall'))} | {fmt(lc.get('bit_accuracy'))} | {fmt(lc.get('payload_or_copyright_recovery'))} | {fmt(lc.get('auth_success'))} |
| unconstrained diffusion re-rendering 200 | {unc.get('sample_count')} | {fmt(unc.get('iou'))} | {fmt(unc.get('dice'))} | {fmt(unc.get('precision'))} | {fmt(unc.get('recall'))} | {fmt(unc.get('bit_accuracy'))} | {fmt(unc.get('payload_or_copyright_recovery'))} | {fmt(unc.get('auth_success'))} |

## Main Conclusion

Pure EditGuard-style localization/copyright extraction is strong for local tamper and localized-composited AIGC edits, but collapses under unconstrained diffusion re-rendering. SCE-LocGuard adds semantic capsule interpretation, report quality calibration, and report-level aggregation on top of this dual-watermark backbone.
"""


def copy_lightweight_outputs(output_dir: Path, results_summary: Path, root_note: Path) -> None:
    results_summary.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("stage10b_*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}:
            shutil.copy2(path, results_summary / path.name)
    shutil.copy2(root_note, results_summary / root_note.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument(
        "--output_dir",
        default="dfg_locguard/outputs/stage10b_editguard_baseline_extraction",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_dir}. Use --overwrite to refresh.")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        SettingSpec(
            key="synthetic_local_tamper_1000",
            display_name="Synthetic local tamper 1000",
            description="Stage9C controlled synthetic local tamper benchmark.",
            output_dir=project_root / "dfg_locguard/outputs/stage9_full_evaluation_1000",
            summary_candidates=["stage9_1000_key_metrics.json", "stage9_summary.json"],
            strategy_candidates=[
                "stage9_1000_strategy_comparison.csv",
                "stage8f/stage8f_strategy_comparison.csv",
            ],
            interpretation="controlled local tamper benchmark",
        ),
        SettingSpec(
            key="localized_composited_aigc_local_edit_1000",
            display_name="Localized-composited AIGC local edit 1000",
            description="Stage10A fair local AIGC edit benchmark preserving unmasked watermarked pixels.",
            output_dir=project_root / "dfg_locguard/outputs/stage10a_lc_scale1000_eval",
            summary_candidates=[
                "stage10a_lc1000_eval_summary.json",
                "stage10a_eval_summary.json",
            ],
            strategy_candidates=[
                "stage10a_lc1000_strategy_comparison.csv",
                "stage10a_strategy_comparison.csv",
                "stage8f/stage8f_strategy_comparison.csv",
            ],
            interpretation="fair local AIGC editing benchmark",
        ),
        SettingSpec(
            key="unconstrained_diffusion_rerendering_200",
            display_name="Unconstrained diffusion re-rendering 200",
            description="Stage10A global re-rendering / watermark laundering stress test.",
            output_dir=project_root / "dfg_locguard/outputs/stage10a_real_aigc_attack_eval",
            summary_candidates=["stage10a_eval_summary.json"],
            strategy_candidates=[
                "stage10a_strategy_comparison.csv",
                "stage8f/stage8f_strategy_comparison.csv",
            ],
            interpretation="global diffusion re-rendering / watermark laundering stress test",
        ),
    ]

    extractions = {spec.key: extract_setting(spec) for spec in specs}
    baseline_rows = [extractions[spec.key]["baseline"] for spec in specs]
    sce_rows: List[Dict[str, Any]] = []
    for spec in specs:
        sce_rows.extend(extractions[spec.key]["sce_rows"])
    capability_rows = make_capability_rows()
    failure_rows = make_failure_rows(extractions)

    summary = {
        "stage": "stage10b_editguard_baseline_extraction",
        "completed": True,
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "git_commit": git_commit_hash(project_root),
        "constraints": {
            "no_training": True,
            "no_vlm": True,
            "no_128bit_payload": True,
            "watermark_embedding_modified": False,
            "robust_payload_bits": 64,
            "previous_outputs_modified": False,
        },
        "editguard_style_baseline": baseline_rows,
        "source_outputs": {
            key: {
                "summary_path": value["summary_path"],
                "strategy_path": value["strategy_path"],
                "per_sample_path": value["baseline"]["source_per_sample"],
            }
            for key, value in extractions.items()
        },
        "external_baseline_feasibility": {
            "omniguard": {
                "status": "feasibility_pending",
                "repo_url": "https://github.com/xuanyuzhang21/OmniGuard",
                "checkpoint_required": True,
                "training_required_for_inference_reproduction": False,
                "notes": "Light inspection shows README checkpoint links and environment.yml; complete training/testing process is still noted as forthcoming.",
            },
            "dual_guard": {
                "status": "conceptual_related_method",
                "repo_url": None,
                "checkpoint_required": "unknown",
                "notes": "No official runnable code/checkpoint confirmed in this stage.",
            },
        },
    }

    write_json(output_dir / "stage10b_editguard_baseline_summary.json", summary)

    baseline_fields = [
        "setting",
        "setting_display_name",
        "sample_count",
        "iou",
        "dice",
        "precision",
        "recall",
        "mae",
        "bit_accuracy",
        "payload_or_copyright_recovery",
        "auth_success",
        "failed_samples",
        "predicted_mask_area_ratio",
        "gt_coverage",
        "source_per_sample",
        "source_summary",
        "notes",
    ]
    write_csv(output_dir / "stage10b_editguard_baseline_by_setting.csv", baseline_rows, baseline_fields)

    sce_fields = [
        "setting",
        "method_variant",
        "sample_count",
        "iou",
        "dice",
        "precision",
        "recall",
        "bit_accuracy",
        "payload_recovery",
        "auth_success",
        "semantic_capsule_recovery",
        "unknown_rate",
        "reports_per_image",
        "gt_coverage_retained",
        "predicted_area_retained",
        "no_report_images",
    ]
    write_csv(output_dir / "stage10b_sce_vs_editguard_numeric_comparison.csv", sce_rows, sce_fields)

    capability_fields = [
        "method",
        "tamper_localization",
        "copyright_provenance_recovery",
        "robust_payload",
        "authentication",
        "semantic_capsule",
        "before_semantics_recovery",
        "after_region_semantic_readout",
        "change_type",
        "severity",
        "report_level_aggregation",
        "compact_audit_level_report",
        "failure_mode_attribution",
        "global_rerendering_stress_test_analysis",
        "status",
    ]
    write_csv(output_dir / "stage10b_method_capability_comparison.csv", capability_rows, capability_fields)

    failure_fields = [
        "setting",
        "iou",
        "dice",
        "bit_accuracy",
        "payload_recovery",
        "auth_success",
        "localization_behavior",
        "payload_auth_behavior",
        "semantic_report_behavior",
        "failure_cause",
        "paper_interpretation",
    ]
    write_csv(
        output_dir / "stage10b_baseline_failure_mode_comparison.csv",
        failure_rows,
        failure_fields,
    )

    overview = build_overview(output_dir, baseline_rows, sce_rows, failure_rows)
    (output_dir / "stage10b_editguard_baseline_overview.md").write_text(
        overview, encoding="utf-8"
    )
    root_note = project_root / "STAGE10B_EDITGUARD_BASELINE_NOTE.md"
    root_note.write_text(build_note(summary), encoding="utf-8")
    copy_lightweight_outputs(
        output_dir,
        project_root / "dfg_locguard" / "results_summary",
        root_note,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
