# Diffusion-Feature-Guided Semantic Localization Watermarking

This repository currently implements **Version 0** of the project.

Version 0 is **not** the final watermarking algorithm. It is used for:

1. building the semantic evaluation pipeline,
2. generating naive TSA/SCC/SSE baselines,
3. visualizing object-level tamper evidence,
4. preparing intermediate data for future diffusion-feature-guided localization watermark training.

The final research direction is:

**Diffusion-Feature-Guided Semantic Localization Watermarking for EditGuard-style Proactive Image Forensics**

The key idea is:

- We do not add a third watermark channel.
- We do not increase watermark payload.
- We keep the robust copyright watermark unchanged.
- We only enhance the fragile localization watermark branch by using frozen diffusion features as training-time semantic change supervision.

During training, frozen diffusion features provide a semantic guidance map for AIGC editing changes.

During inference, no diffusion model is required. The trained EditGuard-style decoder directly outputs copyright bits and a tamper mask.

## Project Structure

```text
dfg_locguard/
  configs/default.yaml
  data/
    original/
    tampered/
    masks/editguard_pred/
    masks/gt/
    metadata.csv
  outputs/
    reports/
    visualizations/
    features/
    results.csv
  src/
    run_v0_pipeline.py
    run_eval.py
    modules/
    dfg/
    watermark/
  scripts/
    demo_v0.py
    cache_diffusion_features.py
    train_locguard_stub.py
```

## Installation

From `D:\pycharm\watermark_exps`:

```powershell
python -m pip install -r dfg_locguard\requirements.txt
```

The first version intentionally keeps dependencies light. SAM, CLIP, DINO, Stable Diffusion, and DIFT are optional future backends.

## Run The CPU Demo

```powershell
python dfg_locguard\scripts\demo_v0.py
```

The demo creates a toy image pair and dummy EditGuard-style mask if no data exists, then runs the full v0 pipeline.

Expected outputs:

```text
dfg_locguard/outputs/results.csv
dfg_locguard/outputs/reports/demo.json
dfg_locguard/outputs/visualizations/demo_vis.png
```

## Use Real EditGuard Masks

For the existing EditGuard workspace, prefer the integrated adapter:

```powershell
python dfg_locguard\scripts\inspect_editguard_dataset.py

python dfg_locguard\scripts\run_editguard_realdata_v0.py `
  --project_root D:\pycharm\watermark_exps `
  --split test `
  --max_samples 20 `
  --region_mode connected_component `
  --mask_source auto
```

This writes:

```text
dfg_locguard/outputs/editguard_realdata/results.csv
dfg_locguard/outputs/editguard_realdata/summary.json
dfg_locguard/outputs/editguard_realdata/failed_cases.json
dfg_locguard/outputs/editguard_realdata/reports/
dfg_locguard/outputs/editguard_realdata/visualizations/
```

Prepare `data/metadata.csv` with:

```csv
image_id,original_path,tampered_path,editguard_mask_path,gt_mask_path,tamper_type,gt_object_label,gt_severity
demo,data/original/demo.png,data/tampered/demo.png,data/masks/editguard_pred/demo.png,data/masks/gt/demo.png,color_replace,object,high
```

Paths may be relative to `dfg_locguard/` or absolute. Then run:

```powershell
python dfg_locguard\src\run_v0_pipeline.py --metadata dfg_locguard\data\metadata.csv
```

The adapter in `src/watermark/editguard_adapter.py` currently supports PNG masks. Later it can be extended to read native EditGuard / OmniGuard outputs.

## Version 0 Modules

- `region_proposal.py`: candidate semantic region generation via `gt_mask`, `connected_components`, `grid`, or optional SAM placeholder.
- `tamper_to_semantic.py`: TSA v0, mapping pixel tamper masks to object-level region scores.
- `semantic_consistency.py`: SCC v0, comparing original/tampered region features. CLIP is optional; fallback uses color histogram and resized pixels.
- `severity_estimation.py`: SSE v0, rule-based severity scoring.
- `visualization.py`: overlay original, tampered, mask, regions, and severity labels.
- `metrics.py`: pixel/object/semantic metric utilities.
- `dfg/`: diffusion feature guidance stubs for future training-time supervision.
- `watermark/`: EditGuard adapter and future localization-watermark training stubs.

## Future Diffusion Feature Guidance

Version 0 does not train a new watermark network.

The future training direction is:

1. Extract frozen Stable Diffusion / DIFT / UNet intermediate features for original and edited images.
2. Build a semantic change guidance map.
3. Use that map only during training to supervise the fragile localization decoder.
4. Keep inference unchanged: the decoder alone predicts copyright bits and tamper mask.
