from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DFG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DFG_ROOT))

from src.modules.common import load_mask_float


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MASK_EXTS = IMAGE_EXTS | {".npy"}
SCAN_DIRS = ["examples", "locwatermark", "valAGE-Set", "valAGE-Set-Mask"]


def inspect_folder(path: Path) -> dict[str, Any]:
    files = sorted([p for p in path.rglob("*") if p.is_file()]) if path.exists() else []
    image_files = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    mask_like_files = [p for p in files if p.suffix.lower() in MASK_EXTS]
    image_examples = []
    for p in image_files[:5]:
        example: dict[str, Any] = {"name": p.name, "relative_path": str(p.relative_to(PROJECT_ROOT))}
        try:
            with Image.open(p) as img:
                example["size"] = list(img.size)
                example["mode"] = img.mode
        except Exception as exc:
            example["error"] = str(exc)
        image_examples.append(example)

    mask_examples = []
    for p in mask_like_files[:5]:
        example = {"name": p.name, "relative_path": str(p.relative_to(PROJECT_ROOT))}
        try:
            arr = load_mask_float(p)
            values = np.unique((arr > 0.5).astype(np.uint8))
            example["shape"] = list(arr.shape)
            example["foreground_ratio"] = float((arr > 0.5).mean())
            example["binary_values_after_threshold"] = values.tolist()
        except Exception as exc:
            example["error"] = str(exc)
        mask_examples.append(example)

    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "image_count": len(image_files),
        "first_10_files": [p.name for p in files[:10]],
        "image_examples": image_examples,
        "mask_examples": mask_examples,
    }


def read_split(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "line_count": 0, "first_20": []}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"path": str(path), "exists": True, "line_count": len(lines), "first_20": lines[:20]}


def stems_in(path: Path, exts: set[str]) -> set[str]:
    if not path.exists():
        return set()
    return {p.stem for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts}


def infer_relationships(dataset_root: Path, split_info: dict[str, Any]) -> dict[str, Any]:
    val_stems = stems_in(dataset_root / "valAGE-Set", IMAGE_EXTS)
    mask_stems = stems_in(dataset_root / "valAGE-Set-Mask", MASK_EXTS)
    example_stems = stems_in(dataset_root / "examples", IMAGE_EXTS)
    loc_stems = stems_in(dataset_root / "locwatermark", IMAGE_EXTS)
    split_stems = {Path(item).stem for item in split_info.get("first_20", [])}
    all_split_stems = set()
    split_path = Path(split_info.get("path", ""))
    if split_path.exists():
        all_split_stems = {Path(line.strip()).stem for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    return {
        "valAGE_Set_and_mask_stem_overlap": len(val_stems & mask_stems),
        "valAGE_Set_count": len(val_stems),
        "valAGE_Set_Mask_count": len(mask_stems),
        "split_samples_in_valAGE_Set": len(all_split_stems & val_stems),
        "split_samples_in_valAGE_Set_Mask": len(all_split_stems & mask_stems),
        "first_20_split_overlap_valAGE_Set": sorted(split_stems & val_stems),
        "first_20_split_overlap_masks": sorted(split_stems & mask_stems),
        "examples_count": len(example_stems),
        "locwatermark_count": len(loc_stems),
        "inference": [
            "valAGE-Set and valAGE-Set-Mask share stems, so they likely pair images with tamper masks.",
            "sep_testlist/sep_vallist entries are image ids used to index valAGE-Set and masks.",
            "locwatermark appears auxiliary unless its stems overlap listed samples.",
            "results/ may contain generated EditGuard masks or previous experiment outputs and should be scanned separately.",
        ],
    }


def scan_result_masks(project_root: Path) -> list[dict[str, Any]]:
    candidates = []
    for base_name in ["results", "tsts", "tests"]:
        base = project_root / base_name
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in MASK_EXTS:
                continue
            lower = p.name.lower()
            parent = p.parent.name.lower()
            if "mask" not in lower and "mask" not in parent and "pred" not in lower:
                continue
            candidates.append({"path": str(p.relative_to(project_root)), "stem": p.stem, "parent": p.parent.name})
    return candidates[:200]


def main() -> None:
    dataset_root = PROJECT_ROOT / "dataset"
    test_split = read_split(dataset_root / "sep_testlist.txt")
    val_split = read_split(dataset_root / "sep_vallist.txt")
    report = {
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(dataset_root),
        "folders": {name: inspect_folder(dataset_root / name) for name in SCAN_DIRS},
        "splits": {
            "test": test_split,
            "val": val_split,
        },
        "relationships": {
            "test": infer_relationships(dataset_root, test_split),
            "val": infer_relationships(dataset_root, val_split),
        },
        "result_mask_candidates": scan_result_masks(PROJECT_ROOT),
    }
    out_path = DFG_ROOT / "outputs" / "dataset_inspection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved inspection: {out_path}")
    print(json.dumps(report["relationships"]["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

