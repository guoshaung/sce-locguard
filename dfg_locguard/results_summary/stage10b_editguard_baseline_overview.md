# Stage 10B-1 EditGuard-Style Baseline Extraction

This stage extracts a pure EditGuard-style baseline from existing outputs only.
It does not train, enable VLMs, expand the 64-bit payload, modify watermark embedding, or rerun large models.

## Pure EditGuard-Style Baseline

| Setting | Samples | IoU | Dice | Precision | Recall | Bit Acc | Payload/Auth |
|---|---:|---:|---:|---:|---:|---:|---:|
| synthetic_local_tamper_1000 | 1000 | 0.8928 | 0.9398 | 0.9210 | 0.9645 | 0.9997 | 0.9890/0.9890 |
| localized_composited_aigc_local_edit_1000 | 1000 | 0.9573 | 0.9778 | 0.9605 | 0.9966 | 0.9997 | 0.9840/0.9840 |
| unconstrained_diffusion_rerendering_200 | 200 | 0.1336 | 0.2304 | 0.1336 | 1.0000 | 0.5058 | 0.0000/0.0000 |

## Interpretation

- Synthetic local tamper is the controlled baseline benchmark.
- Localized-composited AIGC local edit is the fair local AIGC editing benchmark because unmasked watermarked pixels are preserved.
- Unconstrained diffusion re-rendering should be treated as a watermark-laundering stress test.

## Output Files

- `/root/autodl-tmp/watermark_exps/dfg_locguard/outputs/stage10b_editguard_baseline_extraction/stage10b_editguard_baseline_summary.json`
- `/root/autodl-tmp/watermark_exps/dfg_locguard/outputs/stage10b_editguard_baseline_extraction/stage10b_editguard_baseline_by_setting.csv`
- `/root/autodl-tmp/watermark_exps/dfg_locguard/outputs/stage10b_editguard_baseline_extraction/stage10b_method_capability_comparison.csv`
- `/root/autodl-tmp/watermark_exps/dfg_locguard/outputs/stage10b_editguard_baseline_extraction/stage10b_sce_vs_editguard_numeric_comparison.csv`
- `/root/autodl-tmp/watermark_exps/dfg_locguard/outputs/stage10b_editguard_baseline_extraction/stage10b_baseline_failure_mode_comparison.csv`
- `/root/autodl-tmp/watermark_exps/dfg_locguard/outputs/stage10b_editguard_baseline_extraction/stage10b_editguard_baseline_overview.md`

## Failure Mode Summary

| Setting | Failure Cause | Paper Interpretation |
|---|---|---|
| synthetic_local_tamper_1000 | ordinary local tamper and mask fragmentation, not global laundering | primary controlled synthetic benchmark |
| localized_composited_aigc_local_edit_1000 | semantic readout ambiguity in small/tiny edited fragments, not watermark loss | fair local AIGC editing benchmark |
| unconstrained_diffusion_rerendering_200 | global diffusion re-rendering / watermark laundering | stress test, not fair local edit benchmark |
