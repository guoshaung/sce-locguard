from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

CSV_FIELDS = [
    "image_id",
    "gt_mask_path",
    "pred_mask_path",
    "iou",
    "dice",
    "precision",
    "recall",
    "mae",
    "tp",
    "fp",
    "fn",
    "tn",
    "pred_mask_area_ratio",
    "gt_mask_area_ratio",
    "resized_pred",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EditGuard predicted localization masks at pixel level.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--input_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage2_pred_eval")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--contact_sheet_samples", type=int, default=50)
    return parser.parse_args()


def sample_dirs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit())


def load_binary_mask(path: Path, threshold: int, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, NEAREST)
    arr = np.asarray(mask, dtype=np.uint8)
    return arr > threshold


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())
    total = tp + fp + fn + tn
    return {
        "iou": safe_div(tp, tp + fp + fn),
        "dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "mae": safe_div(fp + fn, total),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_mask_area_ratio": safe_div(tp + fp, total),
        "gt_mask_area_ratio": safe_div(tp + fn, total),
    }


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (256, 256)) -> Image.Image:
    header_h = 28
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


def error_map_image(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    out = np.zeros((*gt_bool.shape, 3), dtype=np.uint8)
    out[np.logical_and(pred_bool, gt_bool)] = [255, 255, 255]
    out[np.logical_and(pred_bool, ~gt_bool)] = [255, 80, 80]
    out[np.logical_and(~pred_bool, gt_bool)] = [80, 160, 255]
    return Image.fromarray(out, mode="RGB")


def save_visualization(
    original_path: Path,
    tampered_path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
) -> None:
    panels = [
        title_panel(Image.open(original_path), "original"),
        title_panel(Image.open(tampered_path), "tampered"),
        title_panel(mask_to_image(gt), "gt_mask"),
        title_panel(mask_to_image(pred), "pred_mask"),
        title_panel(error_map_image(gt, pred), "error_map"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def evaluate_sample(sample_dir: Path, output_dir: Path, threshold: int) -> dict[str, Any]:
    image_id = sample_dir.name
    original_path = sample_dir / "original.png"
    tampered_path = sample_dir / "tampered.png"
    gt_mask_path = sample_dir / "gt_mask.png"
    pred_mask_path = sample_dir / "editguard_pred_mask.png"
    row: dict[str, Any] = {
        "image_id": image_id,
        "gt_mask_path": str(gt_mask_path),
        "pred_mask_path": str(pred_mask_path),
        "resized_pred": False,
        "status": "ok",
        "error_message": "",
    }

    try:
        for required in [original_path, tampered_path, gt_mask_path, pred_mask_path]:
            if not required.exists():
                raise FileNotFoundError(f"Missing required file: {required}")

        gt_img = Image.open(gt_mask_path).convert("L")
        pred_img = Image.open(pred_mask_path).convert("L")
        resized_pred = pred_img.size != gt_img.size
        gt = np.asarray(gt_img, dtype=np.uint8) > threshold
        pred = load_binary_mask(pred_mask_path, threshold, size=gt_img.size)
        metrics = compute_metrics(gt, pred)
        row.update(metrics)
        row["resized_pred"] = resized_pred
        save_visualization(original_path, tampered_path, gt, pred, output_dir / "visualizations" / f"{image_id}_eval.png")
    except Exception as exc:
        row.update(
            {
                "iou": "",
                "dice": "",
                "precision": "",
                "recall": "",
                "mae": "",
                "tp": "",
                "fp": "",
                "fn": "",
                "tn": "",
                "pred_mask_area_ratio": "",
                "gt_mask_area_ratio": "",
                "status": "failed",
                "error_message": str(exc),
            }
        )
    return row


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def summarize(rows: list[dict[str, Any]], failed: list[dict[str, str]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in ok_rows]

    return {
        "total_samples": len(rows),
        "evaluated_samples": len(ok_rows),
        "failed_samples": len(failed),
        "mean_iou": mean(values("iou")),
        "mean_dice": mean(values("dice")),
        "mean_precision": mean(values("precision")),
        "mean_recall": mean(values("recall")),
        "mean_mae": mean(values("mae")),
        "median_iou": median(values("iou")),
        "median_dice": median(values("dice")),
        "median_precision": median(values("precision")),
        "median_recall": median(values("recall")),
        "median_mae": median(values("mae")),
    }


def build_contact_sheet(vis_paths: list[Path], output_path: Path, max_samples: int) -> dict[str, Any]:
    if len(vis_paths) > max_samples:
        indices = sorted({round(i * (len(vis_paths) - 1) / (max_samples - 1)) for i in range(max_samples)})
        selected = [vis_paths[index] for index in indices]
        sampling = "even"
    else:
        selected = vis_paths
        sampling = "all"

    rows = []
    for path in selected:
        if path.exists():
            rows.append(Image.open(path).convert("RGB").resize((640, 142), BICUBIC))
    if not rows:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0}

    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(rows)}


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    dirs = sample_dirs(input_dir)
    for idx, sample_dir in enumerate(dirs, start=1):
        row = evaluate_sample(sample_dir, output_dir, args.threshold)
        rows.append(row)
        if row["status"] != "ok":
            failed.append(
                {
                    "image_id": str(row["image_id"]),
                    "failure_stage": "pixel_eval",
                    "error_message": str(row["error_message"]),
                }
            )
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(dirs)} evaluated, failed={len(failed)}")

    per_sample_path = output_dir / "per_sample_metrics.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, failed)
    failed_path = output_dir / "failed_cases.json"
    summary_path = output_dir / "summary_metrics.json"
    failed_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")

    vis_paths = [output_dir / "visualizations" / f"{row['image_id']}_eval.png" for row in rows if row["status"] == "ok"]
    contact_info = build_contact_sheet(vis_paths, output_dir / "stage2_eval_contact_sheet.png", args.contact_sheet_samples)
    summary["contact_sheet"] = contact_info
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved per-sample metrics: {per_sample_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved failed cases: {failed_path}")
    print(f"Saved contact sheet: {output_dir / 'stage2_eval_contact_sheet.png'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
