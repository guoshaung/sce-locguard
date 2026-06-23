from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "image_id",
    "original_path",
    "gt_mask_path",
    "watermarked_path",
    "tampered_path",
    "pred_mask_path",
    "has_original",
    "has_gt_mask",
    "has_watermarked",
    "has_tampered",
    "has_pred_mask",
    "can_eval_pixel",
    "can_run_scc",
    "mask_source",
    "skip_reason",
]

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]
MASK_EXTS = IMAGE_EXTS + [".npy"]

# Output-root convention:
# results/ is the official or legacy EditGuard output tree.
# dfg_locguard/outputs/ is the DFG-LocGuard experiment output tree.


def read_split_ids(dataset_root: Path, split: str) -> list[str]:
    split_file = dataset_root / ("sep_vallist.txt" if split == "val" else "sep_testlist.txt")
    if split_file.exists():
        ids = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                ids.append(Path(value).stem)
        return ids

    image_root = dataset_root / "valAGE-Set"
    return sorted(path.stem for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def find_numbered_file(root: Path, image_id: str, exts: list[str]) -> Path | None:
    return first_existing([root / f"{image_id}{ext}" for ext in exts])


def find_generated_file(generated_root: Path, image_id: str, names: list[str], exts: list[str]) -> Path | None:
    sample_dir = generated_root / image_id
    candidates: list[Path] = []
    for name in names:
        stem = Path(name).stem
        suffix = Path(name).suffix
        if suffix:
            candidates.append(sample_dir / name)
        else:
            candidates.extend(sample_dir / f"{stem}{ext}" for ext in exts)
            candidates.extend(sample_dir / f"{image_id}_{stem}{ext}" for ext in exts)
    return first_existing(candidates)


def find_stage1_output_file(
    generated_root: Path,
    legacy_root: Path | None,
    image_id: str,
    names: list[str],
    exts: list[str],
) -> Path | None:
    generated_path = find_generated_file(generated_root, image_id, names, exts)
    if generated_path is not None:
        return generated_path
    if legacy_root is None:
        return None
    return find_generated_file(legacy_root, image_id, names, exts)


def build_sample_row(
    project_root: Path,
    image_id: str,
    generated_root: Path,
    legacy_root: Path | None,
) -> dict[str, Any]:
    dataset_root = project_root / "dataset"
    # original_path intentionally keeps the existing v0 field name.
    # Here it means the clean host image in dataset/valAGE-Set.
    original_path = find_numbered_file(dataset_root / "valAGE-Set", image_id, IMAGE_EXTS)
    gt_mask_path = find_numbered_file(dataset_root / "valAGE-Set-Mask", image_id, MASK_EXTS)
    watermarked_path = find_stage1_output_file(generated_root, legacy_root, image_id, ["watermarked"], IMAGE_EXTS)
    tampered_path = find_stage1_output_file(generated_root, legacy_root, image_id, ["tampered"], IMAGE_EXTS)
    pred_mask_path = find_stage1_output_file(generated_root, legacy_root, image_id, ["editguard_pred_mask", "pred_mask"], MASK_EXTS)

    has_original = original_path is not None
    has_gt_mask = gt_mask_path is not None
    has_watermarked = watermarked_path is not None
    has_tampered = tampered_path is not None
    has_pred_mask = pred_mask_path is not None
    can_eval_pixel = has_pred_mask and has_gt_mask
    can_run_scc = has_original and has_tampered

    reasons = []
    if not has_original:
        reasons.append("missing_original")
    if not has_gt_mask:
        reasons.append("missing_gt_mask")
    if not has_watermarked:
        reasons.append("missing_watermarked")
    if not has_tampered:
        reasons.append("missing_tampered")
    if not has_pred_mask:
        reasons.append("missing_pred_mask")

    return {
        "image_id": image_id,
        "original_path": str(original_path) if original_path else "",
        "gt_mask_path": str(gt_mask_path) if gt_mask_path else "",
        "watermarked_path": str(watermarked_path) if watermarked_path else "",
        "tampered_path": str(tampered_path) if tampered_path else "",
        "pred_mask_path": str(pred_mask_path) if pred_mask_path else "",
        "has_original": has_original,
        "has_gt_mask": has_gt_mask,
        "has_watermarked": has_watermarked,
        "has_tampered": has_tampered,
        "has_pred_mask": has_pred_mask,
        "can_eval_pixel": can_eval_pixel,
        "can_run_scc": can_run_scc,
        "mask_source": "editguard_pred" if has_pred_mask else "missing",
        "skip_reason": "ok" if not reasons else ";".join(reasons),
    }


def summarize(rows: list[dict[str, Any]], failed: list[dict[str, str]]) -> dict[str, Any]:
    mask_source_counts = Counter(str(row["mask_source"]) for row in rows)
    skip_reason_counts: Counter[str] = Counter()
    for row in rows:
        reasons = str(row["skip_reason"]).split(";")
        for reason in reasons:
            if reason:
                skip_reason_counts[reason] += 1

    return {
        "total_samples": len(rows),
        "has_original_count": sum(bool(row["has_original"]) for row in rows),
        "has_gt_mask_count": sum(bool(row["has_gt_mask"]) for row in rows),
        "has_watermarked_count": sum(bool(row["has_watermarked"]) for row in rows),
        "has_tampered_count": sum(bool(row["has_tampered"]) for row in rows),
        "has_pred_mask_count": sum(bool(row["has_pred_mask"]) for row in rows),
        "can_eval_pixel_count": sum(bool(row["can_eval_pixel"]) for row in rows),
        "can_run_scc_count": sum(bool(row["can_run_scc"]) for row in rows),
        "missing_watermarked_count": sum(not bool(row["has_watermarked"]) for row in rows),
        "missing_tampered_count": sum(not bool(row["has_tampered"]) for row in rows),
        "missing_pred_mask_count": sum(not bool(row["has_pred_mask"]) for row in rows),
        "mask_source_counts": dict(sorted(mask_source_counts.items())),
        "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        "failed_samples": len(failed),
        "failed_sample_details": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DFG-LocGuard Stage 1 data completeness.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--generated_root", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--include_legacy_results", action="store_true")
    parser.add_argument("--legacy_root", default="results/batch_sam_tsa")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/data_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    dataset_root = project_root / "dataset"
    generated_root = (project_root / args.generated_root).resolve()
    legacy_root = (project_root / args.legacy_root).resolve() if args.include_legacy_results else None
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for image_id in read_split_ids(dataset_root, args.split):
        try:
            rows.append(build_sample_row(project_root, image_id, generated_root, legacy_root))
        except Exception as exc:
            failed.append({"image_id": image_id, "reason": str(exc)})

    csv_path = output_dir / "sample_status.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, failed)
    summary_path = output_dir / "data_completeness_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved sample status: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
