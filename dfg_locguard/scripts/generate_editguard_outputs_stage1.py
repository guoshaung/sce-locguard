from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate missing EditGuard Stage 1 outputs without semantic modules.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--image_dir", default="dataset/valAGE-Set")
    parser.add_argument("--mask_dir", default="dataset/valAGE-Set-Mask")
    parser.add_argument("--out_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_editguard_imports(project_root: Path) -> None:
    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def load_ids(dataset_root: Path, split: str, start: int, max_samples: int) -> list[str]:
    split_file = dataset_root / ("sep_vallist.txt" if split == "val" else "sep_testlist.txt")
    if split_file.exists():
        ids = [Path(line.strip()).stem for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        ids = sorted(path.stem for path in (dataset_root / "valAGE-Set").glob("*.png"))
    return ids[start : start + max_samples]


def image_to_editguard_batch(image: np.ndarray) -> dict[str, torch.Tensor]:
    img_gt = image.astype(np.float32) / 255.0
    img_gt = img_gt[:, :, [2, 1, 0]]
    img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float().unsqueeze(0)
    img_gt = torch.nn.functional.interpolate(img_gt, size=(512, 512), mode="nearest", align_corners=None)
    img_gt = img_gt.unsqueeze(0)

    _, t, _, h, w = img_gt.shape
    blue_secret = np.zeros((t, h, w, 3), dtype=np.float32)
    blue_secret[..., 2] = 1.0
    imgs_lq = torch.from_numpy(np.ascontiguousarray(blue_secret)).float().permute(0, 3, 1, 2).unsqueeze(0)
    list_h = torch.stack([imgs_lq], dim=0)
    return {"LQ": list_h, "GT": img_gt}


def load_editguard_model(project_root: Path, opt_path: Path, ckpt_path: Path):
    configure_editguard_imports(project_root)
    import options.options as option
    from models import create_model as create_model_editguard

    opt = option.parse(str(opt_path), is_train=False)
    opt["sdinpaint"] = False
    opt["controlnetinpaint"] = False
    opt["sdxl"] = False
    opt["repaint"] = False
    opt["dist"] = False
    opt = option.dict_to_nonedict(opt)
    torch.backends.cudnn.benchmark = True
    model = create_model_editguard(opt)
    model.load_test(str(ckpt_path))
    return model


def deterministic_tamper(watermarked: np.ndarray, mask: np.ndarray, sample_index: int) -> np.ndarray:
    tampered = watermarked.copy()
    colors = np.asarray(
        [
            [35, 145, 235],
            [235, 90, 60],
            [80, 190, 120],
            [190, 80, 220],
            [245, 185, 45],
        ],
        dtype=np.uint8,
    )
    replacement = np.zeros_like(tampered)
    replacement[:] = colors[sample_index % len(colors)]
    tampered[mask] = (0.25 * tampered[mask] + 0.75 * replacement[mask]).astype(np.uint8)
    return tampered


def draw_stage1_visual(
    host: Image.Image,
    watermarked: Image.Image,
    tampered: Image.Image,
    gt_mask: Image.Image,
    pred_mask: Image.Image,
    output_path: Path,
) -> None:
    thumb_size = (256, 256)
    panels = [
        ("clean host", host.convert("RGB").resize(thumb_size, BICUBIC)),
        ("watermarked", watermarked.convert("RGB").resize(thumb_size, BICUBIC)),
        ("tampered", tampered.convert("RGB").resize(thumb_size, BICUBIC)),
        ("GT tamper mask", gt_mask.convert("L").resize(thumb_size, NEAREST).convert("RGB")),
        ("EditGuard pred mask", pred_mask.convert("L").resize(thumb_size, NEAREST).convert("RGB")),
    ]
    header_h = 28
    canvas = Image.new("RGB", (thumb_size[0] * len(panels), thumb_size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (title, image) in enumerate(panels):
        x = idx * thumb_size[0]
        draw.text((x + 6, 8), title, fill=(20, 20, 20))
        canvas.paste(image, (x, header_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_contact_sheet(sample_dirs: list[Path], output_path: Path, max_items: int = 50) -> dict[str, Any]:
    # A 1000-sample contact sheet would be too tall to inspect comfortably.
    # For large runs, this sheet is an evenly sampled preview capped at max_items.
    if len(sample_dirs) > max_items:
        indices = sorted({round(i * (len(sample_dirs) - 1) / (max_items - 1)) for i in range(max_items)})
        selected_dirs = [sample_dirs[index] for index in indices]
        sampling = "even"
    else:
        selected_dirs = sample_dirs
        sampling = "all"

    rows = []
    for sample_dir in selected_dirs:
        visual_path = sample_dir / "editguard_stage1_vis.png"
        if visual_path.exists():
            rows.append(Image.open(visual_path).convert("RGB").resize((640, 142), BICUBIC))
    if not rows:
        return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": 0, "max_items": max_items}
    canvas = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * row.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {"path": str(output_path.resolve()), "sampling": sampling, "shown_samples": len(rows), "max_items": max_items}


def outputs_complete(sample_dir: Path) -> bool:
    required = [
        "original.png",
        "gt_mask.png",
        "watermarked.png",
        "tampered.png",
        "editguard_pred_mask.png",
        "editguard_stage1_vis.png",
        "meta.json",
    ]
    return all((sample_dir / name).exists() for name in required)


def process_sample(
    model: Any,
    project_root: Path,
    sample_id: str,
    sample_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_path = project_root / args.image_dir / f"{sample_id}.png"
    mask_path = project_root / args.mask_dir / f"{sample_id}.png"
    sample_dir = project_root / args.out_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "image_id": sample_id,
        "sample_dir": str(sample_dir),
        "generated": False,
        "skipped": False,
        "reason": "",
        "failure_stage": "",
        "error_message": "",
        "manual_tamper_pixels": "",
        "editguard_mask_pixels": "",
    }

    if not image_path.exists():
        row.update(
            {
                "skipped": True,
                "reason": "missing_original",
                "failure_stage": "input_validation",
                "error_message": f"Missing clean host image: {image_path}",
            }
        )
        return row
    if not mask_path.exists():
        row.update(
            {
                "skipped": True,
                "reason": "missing_gt_mask",
                "failure_stage": "input_validation",
                "error_message": f"Missing GT tamper mask: {mask_path}",
            }
        )
        return row
    if outputs_complete(sample_dir) and not args.overwrite:
        row.update({"skipped": True, "reason": "already_complete"})
        create_visual_if_possible(sample_dir, image_path, mask_path)
        return row

    host = Image.open(image_path).convert("RGB").resize((512, 512), BICUBIC)
    gt_mask = Image.open(mask_path).convert("L").resize((512, 512), NEAREST)
    edit_mask = np.asarray(gt_mask) > 127
    if int(edit_mask.sum()) == 0:
        row.update(
            {
                "skipped": True,
                "reason": "empty_gt_mask",
                "failure_stage": "input_validation",
                "error_message": f"GT tamper mask has no foreground pixels: {mask_path}",
            }
        )
        return row

    host.save(sample_dir / "original.png")
    Image.fromarray(edit_mask.astype(np.uint8) * 255).save(sample_dir / "gt_mask.png")

    rng = np.random.default_rng(args.seed + int(sample_id) if sample_id.isdigit() else args.seed + sample_index)
    message_bits = rng.choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)

    data = image_to_editguard_batch(np.asarray(host))
    data["MES"] = message_bits
    model.feed_data(data)
    model.mes = message_bits
    watermarked = model.image_hiding()
    Image.fromarray(watermarked).save(sample_dir / "watermarked.png")

    tampered = deterministic_tamper(watermarked, edit_mask, sample_index)
    Image.fromarray(tampered).save(sample_dir / "tampered.png")

    tampered_data = image_to_editguard_batch(tampered)
    tampered_data["MES"] = None
    model.feed_data(tampered_data)
    pred_mask, _ = model.image_recovery(args.threshold)
    pred_mask_binary = np.asarray(pred_mask) > 0
    Image.fromarray(pred_mask_binary.astype(np.uint8) * 255).save(sample_dir / "editguard_pred_mask.png")

    draw_stage1_visual(
        host,
        Image.fromarray(watermarked),
        Image.fromarray(tampered),
        Image.fromarray(edit_mask.astype(np.uint8) * 255),
        Image.fromarray(pred_mask_binary.astype(np.uint8) * 255),
        sample_dir / "editguard_stage1_vis.png",
    )
    meta = {
        "image_id": sample_id,
        "stage": "editguard_stage1_generated",
        "generator": "EditGuard image_hiding + deterministic tamper + image_recovery",
        "original_source_path": str(image_path.resolve()),
        "gt_mask_source_path": str(mask_path.resolve()),
        "original_path": str((sample_dir / "original.png").resolve()),
        "gt_mask_path": str((sample_dir / "gt_mask.png").resolve()),
        "watermarked_path": str((sample_dir / "watermarked.png").resolve()),
        "tampered_path": str((sample_dir / "tampered.png").resolve()),
        "pred_mask_path": str((sample_dir / "editguard_pred_mask.png").resolve()),
        "visualization_path": str((sample_dir / "editguard_stage1_vis.png").resolve()),
        "threshold": args.threshold,
        "seed": args.seed,
        "message_seed": args.seed + int(sample_id) if sample_id.isdigit() else args.seed + sample_index,
        "manual_tamper_pixels": int(edit_mask.sum()),
        "editguard_mask_pixels": int(pred_mask_binary.sum()),
        "notes": [
            "GT mask is used only as the target tamper region and GT localization mask.",
            "GT mask is not used as the predicted mask.",
            "No SAM, CLIP, DINO, or Stable Diffusion module is used in this script.",
            "results/ is the official or legacy output tree; dfg_locguard/outputs/ is the DFG-LocGuard experiment output tree.",
        ],
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    row.update(
        {
            "generated": True,
            "manual_tamper_pixels": int(edit_mask.sum()),
            "editguard_mask_pixels": int(pred_mask_binary.sum()),
        }
    )
    return row


def create_visual_if_possible(sample_dir: Path, image_path: Path, mask_path: Path) -> None:
    try:
        visual_path = sample_dir / "editguard_stage1_vis.png"
        if visual_path.exists():
            return
        original_path = sample_dir / "original.png"
        gt_mask_path = sample_dir / "gt_mask.png"
        draw_stage1_visual(
            Image.open(original_path if original_path.exists() else image_path).convert("RGB").resize((512, 512), BICUBIC),
            Image.open(sample_dir / "watermarked.png").convert("RGB"),
            Image.open(sample_dir / "tampered.png").convert("RGB"),
            Image.open(gt_mask_path if gt_mask_path.exists() else mask_path).convert("L").resize((512, 512), NEAREST),
            Image.open(sample_dir / "editguard_pred_mask.png").convert("L"),
            visual_path,
        )
    except Exception:
        return


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    ckpt_path = project_root / args.ckpt
    opt_path = project_root / args.opt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not opt_path.exists():
        raise FileNotFoundError(f"EditGuard option file not found: {opt_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current EditGuard model configuration.")

    sample_ids = load_ids(project_root / "dataset", args.split, args.start, args.max_samples)
    model = load_editguard_model(project_root, opt_path, ckpt_path)
    rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, str]] = []

    for sample_index, sample_id in enumerate(sample_ids):
        try:
            row = process_sample(model, project_root, sample_id, sample_index, args)
        except Exception as exc:
            row = {
                "image_id": sample_id,
                "sample_dir": str((project_root / args.out_dir / sample_id).resolve()),
                "generated": False,
                "skipped": True,
                "reason": "failed",
                "failure_stage": "process_sample",
                "error_message": str(exc),
                "manual_tamper_pixels": "",
                "editguard_mask_pixels": "",
            }
        if row.get("failure_stage"):
            failed_cases.append(
                {
                    "image_id": str(row["image_id"]),
                    "failure_stage": str(row["failure_stage"]),
                    "error_message": str(row["error_message"]),
                }
            )
        rows.append(row)
        if row["generated"]:
            status = "generated"
        elif row.get("failure_stage"):
            status = f"failed:{row['failure_stage']}"
        else:
            status = f"skipped:{row['reason']}"
        pixels = row["editguard_mask_pixels"]
        print(f"{sample_index + 1:03d}/{len(sample_ids)} {sample_id}: {status} pred_pixels={pixels}")
        if (sample_index + 1) % 100 == 0:
            generated_count = sum(bool(item["generated"]) for item in rows)
            failed_count = len(failed_cases)
            print(f"Progress: {sample_index + 1}/{len(sample_ids)} processed, generated={generated_count}, failed={failed_count}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    out_dir = project_root / args.out_dir
    summary_csv = out_dir / "stage1_generation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "sample_dir",
                "generated",
                "skipped",
                "reason",
                "failure_stage",
                "error_message",
                "manual_tamper_pixels",
                "editguard_mask_pixels",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "selected_samples": len(rows),
        "generated_count": sum(bool(row["generated"]) for row in rows),
        "already_complete_count": sum(row["reason"] == "already_complete" for row in rows),
        "skipped_count": sum(bool(row["skipped"]) for row in rows),
        "failed_count": len(failed_cases),
        "output_dir": str(out_dir.resolve()),
    }
    summary_json = out_dir / "stage1_generation_summary.json"
    failed_cases_json = out_dir / "failed_cases.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    failed_cases_json.write_text(json.dumps(failed_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet = build_contact_sheet([out_dir / sample_id for sample_id in sample_ids], out_dir / "stage1_contact_sheet.png")
    summary["contact_sheet"] = contact_sheet
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved generation CSV: {summary_csv.resolve()}")
    print(f"Saved generation summary: {summary_json.resolve()}")
    print(f"Saved failed cases: {failed_cases_json.resolve()}")
    print(f"Saved contact sheet: {(out_dir / 'stage1_contact_sheet.png').resolve()}")


if __name__ == "__main__":
    main()
