from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import options.options as option
from models import create_model as create_model_editguard
from semantic_guard.io import parse_label_list
from semantic_guard.pipeline import DEFAULT_CATEGORIES, SemanticGuardConfig, SemanticGuardPipeline


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


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


def load_editguard_model(opt_path: Path, ckpt_path: Path):
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


def load_ids(root: Path, list_file: Path, count: int, start: int) -> list[str]:
    with list_file.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    ids = ids[start : start + count]
    return [Path(item).stem for item in ids]


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


def build_contact_sheet(sample_dirs: list[Path], output_path: Path, thumb_size: tuple[int, int] = (192, 192)) -> None:
    labels = ["watermarked", "tampered", "editguard mask", "SAM-TSA overlay"]
    rows = []
    for sample_dir in sample_dirs:
        paths = [
            sample_dir / "watermarked.png",
            sample_dir / "tampered.png",
            sample_dir / "editguard_pred_mask.png",
            sample_dir / "sam_tsa" / "semantic_guard_overlay.png",
        ]
        cells = []
        for label, path in zip(labels, paths):
            img = Image.open(path).convert("RGB").resize(thumb_size, BICUBIC)
            canvas = Image.new("RGB", (thumb_size[0], thumb_size[1] + 28), "white")
            canvas.paste(img, (0, 28))
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 7), label, fill=(20, 20, 20))
            cells.append(canvas)
        row = Image.new("RGB", (thumb_size[0] * len(cells), thumb_size[1] + 28), "white")
        for idx, cell in enumerate(cells):
            row.paste(cell, (idx * thumb_size[0], 0))
        rows.append(row)

    if not rows:
        return
    sheet = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for idx, row in enumerate(rows):
        sheet.paste(row, (0, idx * row.height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch EditGuard mask -> SAM-TSA visualization on official test images.")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--image-dir", default="dataset/valAGE-Set")
    parser.add_argument("--mask-dir", default="dataset/valAGE-Set-Mask")
    parser.add_argument("--list-file", default="dataset/sep_testlist.txt")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--scc-backend", default="histogram", choices=["histogram", "clip", "dino", "hybrid"])
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--out-dir", default="results/batch_sam_tsa")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = load_ids(root, root / args.list_file, args.count, args.start)
    sample_dirs = [out_dir / sample_id for sample_id in sample_ids]

    print(f"Stage 1/2: EditGuard masks for {len(sample_ids)} samples.")
    model = load_editguard_model(root / args.opt, root / args.ckpt)
    rng = np.random.default_rng(20260621)
    rows: list[dict[str, object]] = []

    for idx, sample_id in enumerate(sample_ids):
        sample_dir = out_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = root / args.image_dir / f"{sample_id}.png"
        mask_path = root / args.mask_dir / f"{sample_id}.png"
        image = np.array(Image.open(image_path).convert("RGB").resize((512, 512), BICUBIC))
        edit_mask = np.array(Image.open(mask_path).convert("L").resize((512, 512), NEAREST)) > 127
        if edit_mask.sum() == 0:
            edit_mask[190:330, 190:330] = True

        message_bits = rng.choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)
        data = image_to_editguard_batch(image)
        data["MES"] = message_bits
        model.feed_data(data)
        model.mes = message_bits
        watermarked = model.image_hiding()
        Image.fromarray(watermarked).save(sample_dir / "watermarked.png")

        tampered = deterministic_tamper(watermarked, edit_mask, idx)
        Image.fromarray(tampered).save(sample_dir / "tampered.png")
        Image.fromarray(edit_mask.astype(np.uint8) * 255).save(sample_dir / "manual_tamper_region.png")

        tampered_data = image_to_editguard_batch(tampered)
        tampered_data["MES"] = None
        model.feed_data(tampered_data)
        pred_mask, _ = model.image_recovery(args.threshold)
        pred_mask_binary = np.asarray(pred_mask) > 0
        Image.fromarray(pred_mask_binary.astype(np.uint8) * 255).save(sample_dir / "editguard_pred_mask.png")

        rows.append(
            {
                "sample_id": sample_id,
                "manual_tamper_pixels": int(edit_mask.sum()),
                "editguard_mask_pixels": int(pred_mask_binary.sum()),
            }
        )
        print(f"  {idx + 1:02d}/{len(sample_ids)} {sample_id}: pred_mask_pixels={int(pred_mask_binary.sum())}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Stage 2/2: SAM-TSA overlays.")
    pipeline = SemanticGuardPipeline(
        SemanticGuardConfig(
            tsa_backend="sam",
            scc_backend=args.scc_backend,
            device="cuda",
            sam_checkpoint=str(root / args.sam_checkpoint),
            sam_model_type=args.sam_model_type,
            min_area=32,
            context_dilation=5,
            categories=parse_label_list(args.categories),
        )
    )

    for idx, (sample_id, sample_dir) in enumerate(zip(sample_ids, sample_dirs)):
        report = pipeline.run(
            original_path=sample_dir / "watermarked.png",
            tampered_path=sample_dir / "tampered.png",
            edit_mask_path=sample_dir / "editguard_pred_mask.png",
            output_dir=sample_dir / "sam_tsa",
        )
        objects = report["objects"]
        rows[idx]["objects"] = len(objects)
        best_object = max(objects, key=lambda item: item["sse"]["score"], default=None)
        rows[idx]["max_score"] = best_object["sse"]["score"] if best_object else 0.0
        rows[idx]["max_level"] = best_object["sse"]["level"] if best_object else "none"
        rows[idx]["overlay"] = str(sample_dir / "sam_tsa" / "semantic_guard_overlay.png")
        print(f"  {idx + 1:02d}/{len(sample_ids)} {sample_id}: objects={len(objects)}")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "manual_tamper_pixels",
                "editguard_mask_pixels",
                "objects",
                "max_score",
                "max_level",
                "overlay",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    build_contact_sheet(sample_dirs, out_dir / "contact_sheet.png")
    print(f"Saved summary: {out_dir / 'summary.csv'}")
    print(f"Saved contact sheet: {out_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
