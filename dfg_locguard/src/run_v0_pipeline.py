from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.common import load_rgb, resolve_path
from src.modules.metrics import pixel_metrics
from src.modules.region_proposal import RegionProposal, RegionProposalConfig
from src.modules.semantic_consistency import SCCConfig, SemanticConsistencyChecker
from src.modules.severity_estimation import SSEConfig, SeverityEstimator
from src.modules.tamper_to_semantic import TSAConfig, TamperToSemanticAggregator
from src.modules.visualization import make_visualization
from src.watermark.editguard_adapter import EditGuardAdapter


RESULT_FIELDS = [
    "image_id",
    "region_id",
    "label",
    "bbox",
    "area",
    "tamper_ratio",
    "region_iou",
    "object_tamper_score",
    "semantic_similarity",
    "semantic_change_score",
    "severity_score",
    "severity_level",
    "tamper_type",
    "feature_method",
    "pixel_iou",
    "pixel_f1",
    "report_path",
    "visualization_path",
]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_pipeline(metadata_path: Path, config_path: Path, output_dir: Path, project_root: Path) -> list[dict[str, Any]]:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    vis_dir = output_dir / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    adapter = EditGuardAdapter()
    region_proposal = RegionProposal(RegionProposalConfig(**config.get("region_proposal", {})))
    tsa = TamperToSemanticAggregator(TSAConfig(**config.get("tsa", {})))
    scc_cfg = SCCConfig(**config.get("scc", {}))
    scc = SemanticConsistencyChecker(scc_cfg)
    sse = SeverityEstimator(SSEConfig(**config.get("sse", {})))
    vis_alpha = float(config.get("visualization", {}).get("alpha", 0.45))

    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for item in reader:
            if not item.get("image_id"):
                continue
            image_rows = process_one(
                item=item,
                project_root=project_root,
                output_dir=output_dir,
                reports_dir=reports_dir,
                vis_dir=vis_dir,
                adapter=adapter,
                region_proposal=region_proposal,
                tsa=tsa,
                scc=scc,
                sse=sse,
                vis_alpha=vis_alpha,
            )
            rows.extend(image_rows)

    results_csv = output_dir / "results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def process_one(
    item: dict[str, str],
    project_root: Path,
    output_dir: Path,
    reports_dir: Path,
    vis_dir: Path,
    adapter: EditGuardAdapter,
    region_proposal: RegionProposal,
    tsa: TamperToSemanticAggregator,
    scc: SemanticConsistencyChecker,
    sse: SeverityEstimator,
    vis_alpha: float,
) -> list[dict[str, Any]]:
    image_id = item["image_id"]
    original_path = resolve_path(item.get("original_path"), project_root)
    tampered_path = resolve_path(item.get("tampered_path"), project_root)
    mask_path = resolve_path(item.get("editguard_mask_path"), project_root)
    gt_mask_path = resolve_path(item.get("gt_mask_path"), project_root)
    if original_path is None or tampered_path is None or mask_path is None:
        raise ValueError(f"Missing required paths for image_id={image_id}")

    original = load_rgb(original_path)
    tampered = load_rgb(tampered_path)
    if tampered.size != original.size:
        tampered = tampered.resize(original.size)
    editguard_mask = adapter.load_pred_mask(mask_path, size=original.size)
    gt_mask = adapter.load_pred_mask(gt_mask_path, size=original.size) if gt_mask_path and gt_mask_path.exists() else None

    image_arr = np.asarray(tampered)
    regions = region_proposal.propose(image_arr, editguard_mask, gt_mask_path=gt_mask_path)
    tamper_candidates = tsa.aggregate(editguard_mask, regions)

    triples = []
    row_payloads = []
    pixel = pixel_metrics(editguard_mask, gt_mask) if gt_mask is not None else {}
    for candidate in tamper_candidates:
        semantic = scc.compare(original, tampered, candidate)
        label = item.get("gt_object_label") or candidate.region.label
        severity = sse.estimate(candidate, semantic, label=label)
        triples.append((candidate, semantic, severity))
        row_payloads.append(
            {
                "image_id": image_id,
                "region_id": candidate.region.region_id,
                "label": label,
                "bbox": json.dumps(list(candidate.region.bbox)),
                "area": candidate.region.area,
                "tamper_ratio": candidate.tamper_ratio,
                "region_iou": candidate.region_iou,
                "object_tamper_score": candidate.object_tamper_score,
                "semantic_similarity": semantic.semantic_similarity,
                "semantic_change_score": semantic.semantic_change_score,
                "severity_score": severity.severity_score,
                "severity_level": severity.severity_level,
                "tamper_type": item.get("tamper_type", ""),
                "feature_method": semantic.feature_method,
                "pixel_iou": pixel.get("iou", ""),
                "pixel_f1": pixel.get("f1", ""),
                "report_path": str(reports_dir / f"{image_id}.json"),
                "visualization_path": str(vis_dir / f"{image_id}_vis.png"),
            }
        )

    report = {
        "image_id": image_id,
        "inputs": {
            "original_path": str(original_path),
            "tampered_path": str(tampered_path),
            "editguard_mask_path": str(mask_path),
            "gt_mask_path": str(gt_mask_path) if gt_mask_path else None,
        },
        "tamper_type": item.get("tamper_type", ""),
        "pixel_metrics": pixel,
        "regions": [
            {
                "region": candidate.to_dict(),
                "semantic": semantic.to_dict(),
                "severity": severity.to_dict(),
            }
            for candidate, semantic, severity in triples
        ],
    }
    report_path = reports_dir / f"{image_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    make_visualization(original, tampered, editguard_mask, triples, vis_dir / f"{image_id}_vis.png", alpha=vis_alpha)
    if not row_payloads:
        row_payloads.append(
            {
                "image_id": image_id,
                "region_id": "",
                "label": "",
                "bbox": "",
                "area": "",
                "tamper_ratio": "",
                "region_iou": "",
                "object_tamper_score": "",
                "semantic_similarity": "",
                "semantic_change_score": "",
                "severity_score": "",
                "severity_level": "none",
                "tamper_type": item.get("tamper_type", ""),
                "feature_method": "",
                "pixel_iou": pixel.get("iou", ""),
                "pixel_f1": pixel.get("f1", ""),
                "report_path": str(report_path),
                "visualization_path": str(vis_dir / f"{image_id}_vis.png"),
            }
        )
    return row_payloads


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run Version 0 semantic localization pipeline.")
    parser.add_argument("--metadata", default=str(root / "data" / "metadata.csv"))
    parser.add_argument("--config", default=str(root / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(root / "outputs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata_path = Path(args.metadata).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    project_root = Path(__file__).resolve().parents[1]
    rows = run_pipeline(metadata_path, config_path, output_dir, project_root)
    print(f"Processed rows: {len(rows)}")
    print(f"Saved results: {output_dir / 'results.csv'}")
    print(f"Saved reports: {output_dir / 'reports'}")
    print(f"Saved visualizations: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()

