from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.modules.common import resolve_path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MASK_EXTS = IMAGE_EXTS | {".npy"}
SUFFIXES = [
    "_mask",
    "_gt",
    "_tampered",
    "_watermarked",
    "_loc",
    "_pred",
    "_wm",
    "_editguard_pred_mask",
    "_manual_tamper_region",
]


@dataclass
class EditGuardItem:
    image_id: str
    original_path: Path | None = None
    watermarked_path: Path | None = None
    tampered_path: Path | None = None
    editguard_pred_mask_path: Path | None = None
    gt_mask_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "original_path": self.original_path,
            "watermarked_path": self.watermarked_path,
            "tampered_path": self.tampered_path,
            "editguard_pred_mask_path": self.editguard_pred_mask_path,
            "gt_mask_path": self.gt_mask_path,
            "metadata": self.metadata,
        }


class EditGuardDatasetAdapter:
    """Adapt the existing EditGuard repository data layout to v0 unified items."""

    def __init__(self, project_root: str | Path, split: str = "test", config: dict[str, Any] | None = None):
        self.project_root = Path(project_root).resolve()
        self.split = split
        self.config = config or {}
        self.dataset_root = self.project_root / "dataset"
        self.items: dict[str, EditGuardItem] = {}
        self.warnings: list[dict[str, str]] = []

    def inspect(self) -> dict[str, Any]:
        if not self.items:
            self.build_index()
        index = self.items
        missing_tampered = sum(
            1
            for item in index.values()
            if item.tampered_path is None
            or item.metadata.get("tampered_source") in {"original_as_placeholder", "watermarked_as_placeholder"}
        )
        summary = {
            "project_root": str(self.project_root),
            "split": self.split,
            "total_listed_samples": len(index),
            "successfully_matched_samples": sum(1 for item in index.values() if item.tampered_path and (item.editguard_pred_mask_path or item.gt_mask_path)),
            "missing_original": sum(1 for item in index.values() if item.original_path is None),
            "missing_watermarked": sum(1 for item in index.values() if item.watermarked_path is None),
            "missing_tampered": missing_tampered,
            "missing_mask": sum(1 for item in index.values() if item.editguard_pred_mask_path is None and item.gt_mask_path is None),
            "warnings": self.warnings,
        }
        return summary

    def build_index(self) -> dict[str, EditGuardItem]:
        self.warnings = []
        split_ids = self._read_split_ids()
        originals = self._scan_by_stem(self.dataset_root / "valAGE-Set", IMAGE_EXTS)
        gt_masks = self._scan_by_stem(self.dataset_root / "valAGE-Set-Mask", MASK_EXTS)
        watermarked = self._scan_result_named("watermarked")
        tampered = self._scan_result_named("tampered")
        pred_masks = self._scan_pred_masks()

        self.items = {}
        for image_id in split_ids:
            item = EditGuardItem(image_id=image_id)
            item.original_path = originals.get(image_id)
            item.gt_mask_path = gt_masks.get(image_id)
            item.watermarked_path = watermarked.get(image_id)
            item.tampered_path = tampered.get(image_id)
            item.editguard_pred_mask_path = pred_masks.get(image_id)

            if item.tampered_path is None:
                if item.watermarked_path is not None:
                    item.tampered_path = item.watermarked_path
                    item.metadata["tampered_source"] = "watermarked_as_placeholder"
                    self._warn(image_id, "missing_tampered", "No generated tampered image found; using watermarked image as placeholder.")
                elif item.original_path is not None:
                    item.tampered_path = item.original_path
                    item.metadata["tampered_source"] = "original_as_placeholder"
                    self._warn(image_id, "missing_tampered", "No generated tampered image found; using dataset image as placeholder.")
                else:
                    self._warn(image_id, "missing_tampered", "No tampered image found.")

            if item.editguard_pred_mask_path is None and item.gt_mask_path is not None:
                item.metadata["mask_source"] = "gt_as_placeholder"
            elif item.editguard_pred_mask_path is not None:
                item.metadata["mask_source"] = "editguard_pred"
            else:
                self._warn(image_id, "missing_mask", "No predicted or GT mask found.")

            if item.original_path is None:
                self._warn(image_id, "missing_original", "No original image found; SCC will be skipped.")
            if item.watermarked_path is None:
                self._warn(image_id, "missing_watermarked", "No watermarked image found.")

            self.items[image_id] = item
        return self.items

    def get_item(self, image_id: str) -> EditGuardItem | None:
        if not self.items:
            self.build_index()
        return self.items.get(normalize_stem(image_id))

    def iter_items(self) -> Iterator[EditGuardItem]:
        if not self.items:
            self.build_index()
        yield from self.items.values()

    def save_inspection(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.inspect()
        payload["items_preview"] = [_stringify_item(item) for item in list(self.items.values())[:20]]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_split_ids(self) -> list[str]:
        split_file = self.dataset_root / ("sep_vallist.txt" if self.split == "val" else "sep_testlist.txt")
        if not split_file.exists():
            stems = self._scan_by_stem(self.dataset_root / "valAGE-Set", IMAGE_EXTS)
            return sorted(stems)
        ids = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                ids.append(normalize_stem(Path(line).stem))
        return ids

    def _scan_by_stem(self, root: Path, exts: set[str]) -> dict[str, Path]:
        if not root.exists():
            return {}
        out: dict[str, Path] = {}
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                out.setdefault(normalize_stem(p.stem), p)
        return out

    def _scan_result_named(self, name: str) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for base in [self.project_root / "results", self.project_root / "tsts", self.project_root / "tests"]:
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                    continue
                if p.stem.lower() == name:
                    out.setdefault(normalize_stem(p.parent.name), p)
                elif p.stem.lower().endswith(f"_{name}"):
                    out.setdefault(normalize_stem(p.stem[: -len(name) - 1]), p)
        return out

    def _scan_pred_masks(self) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for base in [self.project_root / "results", self.project_root / "tsts", self.project_root / "tests"]:
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in MASK_EXTS:
                    continue
                lower_name = p.name.lower()
                lower_parent = p.parent.name.lower()
                if "overlay" in lower_name or "semantic" in lower_name:
                    continue
                is_mask = "mask" in lower_name or "mask" in lower_parent or "pred" in lower_name
                if not is_mask:
                    continue
                candidates = [normalize_stem(p.stem), normalize_stem(p.parent.name)]
                if p.stem.lower() in {"editguard_pred_mask", "pred_mask", "mask"}:
                    candidates.insert(0, normalize_stem(p.parent.name))
                for key in candidates:
                    if key and key not in out:
                        out[key] = p
        return out

    def _warn(self, image_id: str, kind: str, message: str) -> None:
        self.warnings.append({"image_id": image_id, "kind": kind, "message": message})


def normalize_stem(stem: str) -> str:
    value = stem.strip()
    changed = True
    while changed:
        changed = False
        lower = value.lower()
        for suffix in SUFFIXES:
            if lower.endswith(suffix):
                value = value[: -len(suffix)]
                changed = True
                break
    return value


def _stringify_item(item: EditGuardItem) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in item.to_dict().items()
    }
