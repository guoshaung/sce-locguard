from __future__ import annotations

import argparse
from pathlib import Path

from semantic_guard.io import parse_label_list
from semantic_guard.pipeline import DEFAULT_CATEGORIES, SemanticGuardConfig, SemanticGuardPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TSA/SCC/SSE on an EditGuard pixel mask.")
    parser.add_argument("--original", required=True, help="Original clean image path.")
    parser.add_argument("--tampered", required=True, help="Tampered image path.")
    parser.add_argument("--edit-mask", required=True, help="EditGuard binary or grayscale tamper mask.")
    parser.add_argument("--out-dir", default="results/semantic_guard", help="Output directory.")
    parser.add_argument("--tsa-backend", default="auto", choices=["auto", "sam", "components"])
    parser.add_argument("--scc-backend", default="auto", choices=["auto", "clip", "dino", "hybrid", "histogram"])
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--sam-checkpoint", default=None, help="Path to SAM checkpoint, e.g. sam_vit_b_01ec64.pth.")
    parser.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--min-area", type=int, default=32)
    parser.add_argument("--context-dilation", type=int, default=5)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--hybrid-clip-weight", type=float, default=0.40)
    parser.add_argument("--hybrid-dino-weight", type=float, default=0.45)
    parser.add_argument("--hybrid-pixel-weight", type=float, default=0.15)
    parser.add_argument("--fusion-model", default=None, help="Optional joblib model trained by train_score_fusion.py.")
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated category labels for optional CLIP zero-shot category weighting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SemanticGuardConfig(
        tsa_backend=args.tsa_backend,
        scc_backend=args.scc_backend,
        device=args.device,
        mask_threshold=args.mask_threshold,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        min_area=args.min_area,
        context_dilation=args.context_dilation,
        clip_model=args.clip_model,
        dino_model=args.dino_model,
        hybrid_clip_weight=args.hybrid_clip_weight,
        hybrid_dino_weight=args.hybrid_dino_weight,
        hybrid_pixel_weight=args.hybrid_pixel_weight,
        fusion_model=args.fusion_model,
        categories=parse_label_list(args.categories),
    )
    report = SemanticGuardPipeline(config).run(
        original_path=args.original,
        tampered_path=args.tampered,
        edit_mask_path=args.edit_mask,
        output_dir=args.out_dir,
    )
    out_dir = Path(args.out_dir)
    print(f"Saved report: {out_dir / 'semantic_guard_report.json'}")
    print(f"Saved overlay: {out_dir / 'semantic_guard_overlay.png'}")
    print(f"Objects: {len(report['objects'])}")
    for item in report["objects"]:
        region = item["region"]
        sse = item["sse"]
        scc = item["scc"]
        print(
            f"  #{region['id']} {region['category']} "
            f"score={sse['score']:.1f} level={sse['level']} "
            f"change={scc['change_score']:.3f} object_ratio={region['object_tamper_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
