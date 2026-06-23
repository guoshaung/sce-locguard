# SemanticGuard: TSA/SCC/SSE Extension for EditGuard

## Current Project Status

This project is based on the existing EditGuard repository structure at `D:\pycharm\watermark_exps`.

Current stage:

**Version 0 only adapts EditGuard-style data to the semantic evaluation pipeline. It does not modify the watermark encoder or decoder yet.**

The long-term target is:

**Diffusion-Feature-Guided Semantic Localization Watermarking.**

Key principle:

- Do not add a third watermark channel.
- Do not increase watermark payload.
- Keep robust copyright watermark unchanged.
- Only enhance the fragile localization watermark branch using training-time semantic guidance from frozen diffusion features.

This workspace keeps the original EditGuard code and adds an object-level semantic post-processing layer:

```text
EditGuard pixel mask
  -> TSA: Tampered Semantic Aggregation
  -> SCC: Semantic Change Checking
  -> SSE: Semantic Severity Estimation
```

## What Changed

- `code/semantic_guard/tsa.py`: groups the EditGuard pixel mask into object-level regions.
  - `components`: connected-component fallback, always runnable.
  - `sam`: optional SAM mask aggregation when `segment-anything` and a SAM checkpoint are provided.
- `code/semantic_guard/scc.py`: compares original/tampered object crops.
  - `clip`: CLIP image embedding similarity and optional zero-shot category label.
  - `dino`: DINOv2 embedding similarity.
  - `histogram`: lightweight fallback for smoke tests.
- `code/semantic_guard/sse.py`: rule-table and weighted severity scoring.
- `code/train_score_fusion.py`: optional logistic/MLP score-fusion trainer.
- `code/semantic_guard_cli.py`: end-to-end CLI.
- `code/maskextract.py`: now accepts input/output path arguments while keeping original defaults.

## GPU Environment

Do not copy the old `editguard_venv`. This machine already reports:

```text
torch 2.6.0+cu124
CUDA available: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

For a fresh venv on this laptop, install CUDA PyTorch first, then the semantic dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements_semantic_guard.txt
```

## Run EditGuard Mask Extraction

After downloading `clean.pth` into `checkpoints/` and the AGE test set into `dataset/valAGE-Set`:

```powershell
cd code
python test.py -opt options/test_editguard.yml --ckpt ../checkpoints/clean.pth
python maskextract.py --threshold 0.2 --input-folder results/test_age-set --output-folder results/mask
```

## Run TSA/SCC/SSE

Fast smoke test without SAM or Stable Diffusion downloads:

```powershell
python code/smoke_semantic_guard.py
```

This writes a one-image end-to-end result to `results/smoke_semantic_guard`.

Batch SAM-TSA visualization on official test images:

```powershell
python code/batch_sam_tsa_visualize.py --count 8 --sam-checkpoint checkpoints\sam_vit_b_01ec64.pth
```

This writes per-sample overlays and a contact sheet to `results/batch_sam_tsa`.

## Real EditGuard Data Adapter

The integrated research prototype lives in:

```text
dfg_locguard/
```

Output-root convention:

- `results/` is the official EditGuard or legacy experiment output directory.
- `dfg_locguard/outputs/` is the DFG-LocGuard experiment output directory.
- Stage 1 EditGuard-only generated outputs are written to `dfg_locguard/outputs/editguard_stage1_generated/`.

It adapts the current EditGuard dataset layout:

```text
dataset/examples
dataset/locwatermark
dataset/valAGE-Set
dataset/valAGE-Set-Mask
dataset/sep_testlist.txt
dataset/sep_vallist.txt
results/
```

Inspect the current dataset structure:

```powershell
python dfg_locguard\scripts\inspect_editguard_dataset.py
```

This writes:

```text
dfg_locguard/outputs/dataset_inspection.json
```

Run Stage 1 data completeness audit:

```powershell
python dfg_locguard\scripts\audit_data_completeness.py `
  --project_root D:\pycharm\watermark_exps `
  --split test
```

Generate EditGuard-only Stage 1 outputs without SAM/CLIP/DINO/Stable Diffusion:

```powershell
python dfg_locguard\scripts\generate_editguard_outputs_stage1.py `
  --project_root D:\pycharm\watermark_exps `
  --split test `
  --max_samples 20
```

Run v0 semantic evaluation on real EditGuard-style data:

```powershell
python dfg_locguard\scripts\run_editguard_realdata_v0.py `
  --project_root D:\pycharm\watermark_exps `
  --split test `
  --max_samples 20 `
  --region_mode connected_component `
  --mask_source auto
```

Outputs:

```text
dfg_locguard/outputs/editguard_realdata/reports/
dfg_locguard/outputs/editguard_realdata/visualizations/
dfg_locguard/outputs/editguard_realdata/results.csv
dfg_locguard/outputs/editguard_realdata/summary.json
dfg_locguard/outputs/editguard_realdata/failed_cases.json
```

If no real EditGuard predicted mask is found, the adapter can use `valAGE-Set-Mask` as a placeholder predicted mask and marks it as:

```text
mask_source = gt_as_placeholder
```

Use hybrid CLIP+DINO SCC for the final semantic-change setting:

```powershell
python code/batch_sam_tsa_visualize.py `
  --count 8 `
  --sam-checkpoint checkpoints\sam_vit_b_01ec64.pth `
  --scc-backend hybrid `
  --out-dir results\batch_sam_tsa_hybrid
```

Use the lightweight fallback first:

```powershell
python code/semantic_guard_cli.py `
  --original dataset/examples/0011.png `
  --tampered path\to\tampered.png `
  --edit-mask path\to\editguard_mask.png `
  --out-dir results/semantic_guard_demo `
  --tsa-backend components `
  --scc-backend histogram `
  --device cuda
```

Use SAM + CLIP when checkpoints/models are available:

```powershell
python code/semantic_guard_cli.py `
  --original path\to\original.png `
  --tampered path\to\tampered.png `
  --edit-mask path\to\editguard_mask.png `
  --out-dir results/semantic_guard_sam_clip `
  --tsa-backend sam `
  --sam-checkpoint checkpoints\sam_vit_b_01ec64.pth `
  --sam-model-type vit_b `
  --scc-backend clip `
  --device cuda
```

Outputs:

- `semantic_guard_report.json`
- `semantic_guard_overlay.png`
- `semantic_objects.png`
- `object_XXX_mask.png`

## Optional Score Fusion

Prepare a CSV with:

```text
object_tamper_ratio,image_tamper_ratio,semantic_change,category_weight,tsa_confidence,pixel_delta,label
```

Then train:

```powershell
python code/train_score_fusion.py --csv data\severity_labels.csv --out checkpoints\semantic_guard_score_fusion.joblib --model logistic
```

Use it during inference:

```powershell
python code/semantic_guard_cli.py ... --fusion-model checkpoints\semantic_guard_score_fusion.joblib
```

## Paper Angle

The extension moves EditGuard from pixel-level localization to object-level forensic interpretation:

- TSA reduces scattered watermark residuals into semantically meaningful tampered entities.
- SCC separates harmless texture/noise changes from object-level semantic replacement.
- SSE converts localization and semantic evidence into a calibrated severity score.

This gives a clean ablation path for a CCF-C paper: EditGuard mask only, plus TSA, plus SCC, plus SSE, plus optional learned score fusion.
