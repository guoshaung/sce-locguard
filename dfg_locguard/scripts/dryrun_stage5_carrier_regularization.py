from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run Stage 5 fragile carrier regularization hooks.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--input_dir", default="dfg_locguard/outputs/editguard_stage1_generated")
    parser.add_argument("--output_dir", default="dfg_locguard/outputs/stage5_training_regularization/dryrun")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--lambda_sparse", type=float, default=0.0)
    parser.add_argument("--lambda_bg", type=float, default=0.0)
    parser.add_argument("--lambda_floor", type=float, default=0.0)
    parser.add_argument("--lambda_sem", type=float, default=0.0)
    parser.add_argument("--residual_floor_tau", type=float, default=0.01)
    parser.add_argument("--recovery_threshold", type=float, default=0.2)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--seed", type=int, default=20260621)
    return parser.parse_args()


def configure_editguard_imports(project_root: Path) -> None:
    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def load_editguard_model(project_root: Path, opt_path: Path, ckpt_path: Path, args: argparse.Namespace):
    configure_editguard_imports(project_root)
    import options.options as option
    from models import create_model as create_model_editguard

    opt = option.parse(str(opt_path), is_train=False)
    opt["sdinpaint"] = False
    opt["controlnetinpaint"] = False
    opt["sdxl"] = False
    opt["repaint"] = False
    opt["dist"] = False
    opt.setdefault("train", {})
    opt["train"]["enable_carrier_regularization"] = True
    opt["train"]["lambda_sparse"] = args.lambda_sparse
    opt["train"]["lambda_bg"] = args.lambda_bg
    opt["train"]["lambda_floor"] = args.lambda_floor
    opt["train"]["lambda_sem"] = args.lambda_sem
    opt["train"]["residual_floor_tau"] = args.residual_floor_tau
    opt["train"]["return_fragile_carrier"] = True
    opt = option.dict_to_nonedict(opt)
    model = create_model_editguard(opt)
    model.load_test(str(ckpt_path))
    return model


def sample_dirs(input_dir: Path, start: int, max_samples: int) -> list[Path]:
    dirs = sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.isdigit())
    return dirs[start : start + max_samples]


def image_to_editguard_batch(image: np.ndarray, gt_mask: np.ndarray | None = None) -> dict[str, torch.Tensor]:
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
    data = {"LQ": list_h, "GT": img_gt}
    if gt_mask is not None:
        data["GT_MASK"] = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return data


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


def tensor_shape(tensor: torch.Tensor | None) -> list[int] | None:
    return list(tensor.shape) if tensor is not None else None


def heatmap_from_residual(residual: torch.Tensor) -> Image.Image:
    residual_map = residual.detach().float().abs().mean(dim=1)[0].cpu().numpy()
    residual_map = residual_map - float(residual_map.min())
    residual_map = residual_map / max(float(residual_map.max()), 1e-12)
    heat = np.clip(residual_map * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heat, mode="RGB")


def title_panel(image: Image.Image, title: str, size: tuple[int, int] = (220, 220)) -> Image.Image:
    header_h = 28
    image = image.convert("RGB").resize(size, BICUBIC)
    canvas = Image.new("RGB", (size[0], size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill=(20, 20, 20))
    canvas.paste(image, (0, header_h))
    return canvas


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")


def build_residual_visual(rows: list[dict[str, Any]], output_path: Path) -> None:
    panels = []
    for row in rows:
        panels.extend(
            [
                title_panel(Image.open(row["original_path"]), f"{row['image_id']} original"),
                title_panel(mask_image(np.asarray(Image.open(row["gt_mask_path"]).convert("L")) > 127), "gt mask"),
                title_panel(row["residual_heatmap"], "abs residual"),
            ]
        )
    if not panels:
        return
    row_width = panels[0].width * 3
    row_height = panels[0].height
    canvas = Image.new("RGB", (row_width, row_height * (len(panels) // 3)), "white")
    for idx, panel in enumerate(panels):
        x = (idx % 3) * panel.width
        y = (idx // 3) * panel.height
        canvas.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_editguard_model(project_root, project_root / args.opt, project_root / args.ckpt, args)
    dryrun_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample_index, sample_dir in enumerate(sample_dirs(input_dir, args.start, args.max_samples)):
            image_id = sample_dir.name
            original_path = sample_dir / "original.png"
            gt_mask_path = sample_dir / "gt_mask.png"
            original = Image.open(original_path).convert("RGB").resize((512, 512), BICUBIC)
            gt_mask = np.asarray(Image.open(gt_mask_path).convert("L").resize((512, 512), NEAREST), dtype=np.uint8) > args.mask_threshold
            rng_seed = args.seed + int(image_id) if image_id.isdigit() else args.seed + sample_index
            rng = np.random.default_rng(rng_seed)
            message_bits = rng.choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)

            data = image_to_editguard_batch(np.asarray(original), gt_mask)
            data["MES"] = message_bits
            model.feed_data(data)
            model.mes = message_bits
            watermarked, carrier_payload = model.image_hiding(return_fragile_carrier=True)
            losses = model.compute_carrier_regularization_losses(
                carrier_payload["residual"],
                gt_mask=model.gt_mask,
                semantic_map=None,
            )

            tampered = deterministic_tamper(watermarked, gt_mask, sample_index)
            recovery_data = image_to_editguard_batch(tampered, gt_mask)
            recovery_data["MES"] = None
            model.feed_data(recovery_data)
            pred_mask, recmessage = model.image_recovery(args.recovery_threshold)

            dryrun_rows.append(
                {
                    "image_id": image_id,
                    "carrier_sparsity_loss": float(losses["carrier_sparsity_loss"].detach().cpu()),
                    "background_residual_suppression_loss": float(losses["background_residual_suppression_loss"].detach().cpu()),
                    "residual_floor_loss": float(losses["residual_floor_loss"].detach().cpu()),
                    "semantic_guided_residual_loss": float(losses["semantic_guided_residual_loss"].detach().cpu()),
                    "carrier_regularization_total": float(losses["carrier_regularization_total"].detach().cpu()),
                    "watermarked_shape": list(watermarked.shape),
                    "pred_mask_shape": list(np.asarray(pred_mask).shape),
                    "pred_mask_pixels": int((np.asarray(pred_mask) > 0).sum()),
                    "recovered_message_shape": tensor_shape(recmessage),
                }
            )
            shape_rows.append(
                {
                    "image_id": image_id,
                    "LQ": tensor_shape(data["LQ"]),
                    "GT": tensor_shape(data["GT"]),
                    "GT_MASK": tensor_shape(data["GT_MASK"]),
                    "fragile_carrier": tensor_shape(carrier_payload["fragile_carrier"]),
                    "carrier_for_bitencoder": tensor_shape(carrier_payload["carrier_for_bitencoder"]),
                    "encoded_image": tensor_shape(carrier_payload["encoded_image"]),
                    "host_image": tensor_shape(carrier_payload["host_image"]),
                    "residual": tensor_shape(carrier_payload["residual"]),
                }
            )
            visual_rows.append(
                {
                    "image_id": image_id,
                    "original_path": str(original_path.resolve()),
                    "gt_mask_path": str(gt_mask_path.resolve()),
                    "residual_heatmap": heatmap_from_residual(carrier_payload["residual"]),
                }
            )

    mean_losses = {
        key: float(np.mean([row[key] for row in dryrun_rows])) if dryrun_rows else 0.0
        for key in [
            "carrier_sparsity_loss",
            "background_residual_suppression_loss",
            "residual_floor_loss",
            "semantic_guided_residual_loss",
            "carrier_regularization_total",
        ]
    }
    loss_payload = {
        "stage": "stage5_carrier_regularization_dryrun",
        "samples": dryrun_rows,
        "mean_losses": mean_losses,
        "regularization_weights": {
            "lambda_sparse": args.lambda_sparse,
            "lambda_bg": args.lambda_bg,
            "lambda_floor": args.lambda_floor,
            "lambda_sem": args.lambda_sem,
            "residual_floor_tau": args.residual_floor_tau,
        },
        "notes": [
            "No optimizer.step() is called.",
            "GT mask is used only to compute optional training-time regularization losses.",
            "GT mask is not used as a predicted mask.",
            "No Stable Diffusion, SAM, CLIP, or DINO module is used.",
        ],
    }
    (output_dir / "dryrun_loss_values.json").write_text(json.dumps(loss_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dryrun_tensor_shapes.json").write_text(
        json.dumps({"stage": "stage5_carrier_regularization_dryrun", "samples": shape_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_residual_visual(visual_rows, output_dir / "dryrun_residual_visualization.png")

    print(f"Saved loss values: {output_dir / 'dryrun_loss_values.json'}")
    print(f"Saved tensor shapes: {output_dir / 'dryrun_tensor_shapes.json'}")
    print(f"Saved residual visualization: {output_dir / 'dryrun_residual_visualization.png'}")
    print(json.dumps(loss_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
