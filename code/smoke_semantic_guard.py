from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import options.options as option
from models import create_model as create_model_editguard
from semantic_guard.pipeline import SemanticGuardConfig, SemanticGuardPipeline


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast one-image smoke test for EditGuard + SemanticGuard.")
    parser.add_argument("--image", default="dataset/valAGE-Set/0001.png")
    parser.add_argument("--tamper-mask", default="dataset/valAGE-Set-Mask/0001.png")
    parser.add_argument("--ckpt", default="checkpoints/clean.pth")
    parser.add_argument("--opt", default="code/options/test_editguard.yml")
    parser.add_argument("--out-dir", default="results/smoke_semantic_guard")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_editguard_model(root / args.opt, root / args.ckpt)

    image = np.array(Image.open(root / args.image).convert("RGB").resize((512, 512), Image.BICUBIC))
    message_bits = np.random.default_rng(1234).choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)

    data = image_to_editguard_batch(image)
    data["MES"] = message_bits
    model.feed_data(data)
    model.mes = message_bits
    watermarked = model.image_hiding()
    Image.fromarray(watermarked).save(out_dir / "0001_watermarked.png")

    edit_mask = np.array(Image.open(root / args.tamper_mask).convert("L").resize((512, 512), Image.NEAREST)) > 127
    if edit_mask.sum() == 0:
        edit_mask[190:330, 190:330] = True

    tampered = watermarked.copy()
    replacement = np.zeros_like(tampered)
    replacement[..., 0] = 35
    replacement[..., 1] = 145
    replacement[..., 2] = 235
    tampered[edit_mask] = (0.2 * tampered[edit_mask] + 0.8 * replacement[edit_mask]).astype(np.uint8)
    Image.fromarray(tampered).save(out_dir / "0001_tampered.png")
    Image.fromarray(edit_mask.astype(np.uint8) * 255).save(out_dir / "0001_manual_tamper_region.png")

    tampered_data = image_to_editguard_batch(tampered)
    tampered_data["MES"] = None
    model.feed_data(tampered_data)
    pred_mask, _ = model.image_recovery(args.threshold)
    pred_mask_binary = np.asarray(pred_mask) > 0
    Image.fromarray(pred_mask_binary.astype(np.uint8) * 255).save(out_dir / "0001_editguard_pred_mask.png")

    report = SemanticGuardPipeline(
        SemanticGuardConfig(
            tsa_backend="components",
            scc_backend="histogram",
            device="cuda",
            min_area=32,
            context_dilation=5,
            categories=[],
        )
    ).run(
        original_path=out_dir / "0001_watermarked.png",
        tampered_path=out_dir / "0001_tampered.png",
        edit_mask_path=out_dir / "0001_editguard_pred_mask.png",
        output_dir=out_dir / "semantic_guard",
    )

    print(f"saved_dir: {out_dir}")
    print(f"manual_tamper_pixels: {int(edit_mask.sum())}")
    print(f"pred_mask_pixels: {int(pred_mask_binary.sum())}")
    print(f"objects: {len(report['objects'])}")
    for item in report["objects"][:5]:
        region = item["region"]
        scc = item["scc"]
        sse = item["sse"]
        print(
            f"#{region['id']} bbox={region['bbox_xyxy']} "
            f"change={scc['change_score']:.4f} score={sse['score']:.2f} level={sse['level']}"
        )


if __name__ == "__main__":
    main()
