from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

DFG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DFG_ROOT))

from src.adapters.editguard_dataset_adapter import EditGuardDatasetAdapter, EditGuardItem
from src.modules.common import SemanticResult, load_mask, load_rgb
from src.modules.region_proposal import RegionProposal, RegionProposalConfig
from src.modules.semantic_consistency import SCCConfig, SemanticConsistencyChecker
from src.modules.severity_estimation import SSEConfig, SeverityEstimator
from src.modules.tamper_to_semantic import TSAConfig, TamperToSemanticAggregator
from src.modules.visualization import make_visualization


RESULT_FIELDS = [
    "image_id",
    "region_id",
    "bbox",
    "area",
    "label",
    "tamper_ratio",
    "region_iou",
    "object_tamper_score",
    "semantic_similarity",
    "semantic_change_score",
    "severity_score",
    "severity_level",
    "mask_source",
    "region_mode",
    "scc_status",
    "original_path",
    "tampered_path",
    "mask_path",
]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def normalize_region_mode(mode: str) -> str:
    aliases = {
        "connected_component": "connected_components",
        "connected_components": "connected_components",
        "component": "connected_components",
        "gt": "gt_mask",
        "gt_mask": "gt_mask",
        "grid": "grid",
        "sam": "sam",
    }
    return aliases.get(mode, mode)


def choose_mask(item: EditGuardItem, mask_source: str) -> tuple[Path | None, str]:
    if mask_source == "editguard_pred":
        return item.editguard_pred_mask_path, "editguard_pred"
    if mask_source == "gt_as_placeholder":
        return item.gt_mask_path, "gt_as_placeholder"
    if item.editguard_pred_mask_path is not None:
        return item.editguard_pred_mask_path, "editguard_pred"
    if item.gt_mask_path is not None:
        return item.gt_mask_path, "gt_as_placeholder"
    return None, "missing"


def process_item(
    item: EditGuardItem,
    region_proposal: RegionProposal,
    tsa: TamperToSemanticAggregator,
    scc: SemanticConsistencyChecker,
    sse: SeverityEstimator,
    output_dir: Path,
    region_mode: str,
    mask_source_arg: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    mask_path, actual_mask_source = choose_mask(item, mask_source_arg)
    if item.tampered_path is None and item.original_path is None and item.watermarked_path is None:
        return [], {"image_id": item.image_id, "reason": "missing_image"}
    if mask_path is None:
        return [], {"image_id": item.image_id, "reason": "missing_mask"}

    display_path = item.tampered_path or item.watermarked_path or item.original_path
    assert display_path is not None
    tampered = load_rgb(display_path)
    original = load_rgb(item.original_path) if item.original_path is not None else None
    editguard_mask = load_mask(mask_path, size=tampered.size)

    gt_mask_path = item.gt_mask_path if region_mode == "gt_mask" else None
    regions = region_proposal.propose(np.asarray(tampered), editguard_mask, gt_mask_path=gt_mask_path)
    candidates = tsa.aggregate(editguard_mask, regions)

    triples = []
    rows: list[dict[str, Any]] = []
    scc_status = "ok"
    if original is None:
        scc_status = "skipped_no_original"
    elif item.metadata.get("tampered_source") in {"original_as_placeholder", "watermarked_as_placeholder"}:
        scc_status = "skipped_no_tampered"
    elif original.size != tampered.size:
        original = original.resize(tampered.size)

    for candidate in candidates:
        if scc_status == "ok" and original is not None:
            semantic = scc.compare(original, tampered, candidate)
            semantic_similarity: float | str = semantic.semantic_similarity
            semantic_change: float | str = semantic.semantic_change_score
        else:
            semantic = SemanticResult(1.0, 0.0, scc_status)
            semantic_similarity = ""
            semantic_change = ""
        label = candidate.region.label
        severity = sse.estimate(candidate, semantic, label=label)
        triples.append((candidate, semantic, severity))
        rows.append(
            {
                "image_id": item.image_id,
                "region_id": candidate.region.region_id,
                "bbox": json.dumps(list(candidate.region.bbox)),
                "area": candidate.region.area,
                "label": label,
                "tamper_ratio": candidate.tamper_ratio,
                "region_iou": candidate.region_iou,
                "object_tamper_score": candidate.object_tamper_score,
                "semantic_similarity": semantic_similarity,
                "semantic_change_score": semantic_change,
                "severity_score": severity.severity_score,
                "severity_level": severity.severity_level,
                "mask_source": actual_mask_source,
                "region_mode": region_mode,
                "scc_status": scc_status,
                "original_path": str(item.original_path) if item.original_path else "",
                "tampered_path": str(display_path),
                "mask_path": str(mask_path),
            }
        )

    report = {
        "image_id": item.image_id,
        "paths": {
            "original_path": str(item.original_path) if item.original_path else None,
            "watermarked_path": str(item.watermarked_path) if item.watermarked_path else None,
            "tampered_path": str(display_path),
            "editguard_pred_mask_path": str(item.editguard_pred_mask_path) if item.editguard_pred_mask_path else None,
            "gt_mask_path": str(item.gt_mask_path) if item.gt_mask_path else None,
            "used_mask_path": str(mask_path),
        },
        "metadata": item.metadata,
        "mask_source": actual_mask_source,
        "region_mode": region_mode,
        "scc_status": scc_status,
        "mask_foreground_ratio": float(editguard_mask.mean()),
        "candidate_regions": len(regions),
        "selected_regions": [
            {"region": candidate.to_dict(), "semantic": semantic.to_dict(), "severity": severity.to_dict()}
            for candidate, semantic, severity in triples
        ],
    }
    reports_dir = output_dir / "reports"
    vis_dir = output_dir / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{item.image_id}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    vis_original = original if original is not None else tampered
    make_visualization(vis_original, tampered, editguard_mask, triples, vis_dir / f"{item.image_id}_vis.png")
    return rows, None


def summarize(
    rows: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    image_sizes: list[tuple[int, int]],
    mask_ratios: list[float],
    candidate_counts: list[int],
    total_samples: int,
    region_mode: str,
    mask_source: str,
    feature_method: str,
    adapter_summary: dict[str, Any],
) -> dict[str, Any]:
    severity_distribution: dict[str, int] = {}
    mask_source_distribution: dict[str, int] = {}
    scc_status_distribution: dict[str, int] = {}
    for row in rows:
        level = str(row.get("severity_level", "none"))
        severity_distribution[level] = severity_distribution.get(level, 0) + 1
        row_mask_source = str(row.get("mask_source", "unknown"))
        mask_source_distribution[row_mask_source] = mask_source_distribution.get(row_mask_source, 0) + 1
        row_scc_status = str(row.get("scc_status", "unknown"))
        scc_status_distribution[row_scc_status] = scc_status_distribution.get(row_scc_status, 0) + 1
    avg_size = [
        float(np.mean([size[0] for size in image_sizes])) if image_sizes else 0.0,
        float(np.mean([size[1] for size in image_sizes])) if image_sizes else 0.0,
    ]
    return {
        "total_listed_samples": adapter_summary.get("total_listed_samples", total_samples),
        "successfully_matched_samples": adapter_summary.get("successfully_matched_samples", 0),
        "missing_original": adapter_summary.get("missing_original", 0),
        "missing_watermarked": adapter_summary.get("missing_watermarked", 0),
        "missing_tampered": adapter_summary.get("missing_tampered", 0),
        "missing_mask": adapter_summary.get("missing_mask", 0),
        "total_samples": total_samples,
        "processed_samples": len({row["image_id"] for row in rows}),
        "failed_samples": len(failed),
        "result_rows": len(rows),
        "average_image_size": avg_size,
        "average_mask_foreground_ratio": float(np.mean(mask_ratios)) if mask_ratios else 0.0,
        "average_candidate_regions": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "severity_distribution": severity_distribution,
        "severity_distribution_level": "region",
        "mask_source_distribution": mask_source_distribution,
        "scc_status_distribution": scc_status_distribution,
        "region_mode": region_mode,
        "mask_source": mask_source,
        "feature_method": feature_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dfg_locguard v0 on existing EditGuard project data.")
    parser.add_argument("--project_root", default=str(DFG_ROOT.parents[0]))
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--region_mode", default="connected_component")
    parser.add_argument("--mask_source", default="auto", choices=["auto", "editguard_pred", "gt_as_placeholder"])
    parser.add_argument("--config", default=str(DFG_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output_dir", default=str(DFG_ROOT / "outputs" / "editguard_realdata"))
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    region_mode = normalize_region_mode(args.region_mode)

    adapter = EditGuardDatasetAdapter(project_root, split=args.split)
    adapter.build_index()
    adapter_summary = adapter.inspect()
    adapter.save_inspection(output_dir / "adapter_inspection.json")
    items = list(adapter.iter_items())[: args.max_samples]

    region_config = RegionProposalConfig(**config.get("region_proposal", {}))
    region_config.mode = region_mode
    tsa = TamperToSemanticAggregator(TSAConfig(**config.get("tsa", {})))
    scc_config = SCCConfig(**config.get("scc", {}))
    scc = SemanticConsistencyChecker(scc_config)
    sse = SeverityEstimator(SSEConfig(**config.get("sse", {})))
    region_proposal = RegionProposal(region_config)

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    image_sizes: list[tuple[int, int]] = []
    mask_ratios: list[float] = []
    candidate_counts: list[int] = []

    for item in items:
        try:
            item_rows, failure = process_item(
                item,
                region_proposal=region_proposal,
                tsa=tsa,
                scc=scc,
                sse=sse,
                output_dir=output_dir,
                region_mode=region_mode,
                mask_source_arg=args.mask_source,
            )
            if failure:
                failed.append(failure)
                continue
            rows.extend(item_rows)
            report = json.loads((output_dir / "reports" / f"{item.image_id}.json").read_text(encoding="utf-8"))
            if item.tampered_path:
                with Image.open(item.tampered_path) as img:
                    image_sizes.append(img.size)
            mask_ratios.append(float(report.get("mask_foreground_ratio", 0.0)))
            candidate_counts.append(int(report.get("candidate_regions", 0)))
        except Exception as exc:
            failed.append({"image_id": item.image_id, "reason": str(exc)})

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(
        rows,
        failed,
        image_sizes,
        mask_ratios,
        candidate_counts,
        total_samples=len(items),
        region_mode=region_mode,
        mask_source=args.mask_source,
        feature_method=scc_config.feature_method,
        adapter_summary=adapter_summary,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "failed_cases.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed samples: {summary['processed_samples']}/{summary['total_samples']}")
    print(f"Failed samples: {summary['failed_samples']}")
    print(f"Saved results: {output_dir / 'results.csv'}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
