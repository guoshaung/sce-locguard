# Stage 10B EditGuard Baseline Note

## Scope

- Pure EditGuard-style baseline extraction from existing Stage9C / Stage10A outputs.
- No training.
- No VLM.
- No 128-bit payload.
- No watermark embedding modification.
- Robust payload remains 64-bit.
- Stage8E/Stage8F semantic report metrics are excluded from the pure baseline.

## Pure EditGuard-Style Baseline Metrics

| Setting | Samples | IoU | Dice | Precision | Recall | Bit Acc | Payload Recovery | Auth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic local tamper 1000 | 1000 | 0.89283 | 0.93982 | 0.92101 | 0.96452 | 0.99972 | 0.98900 | 0.98900 |
| localized-composited AIGC local edit 1000 | 1000 | 0.95732 | 0.97783 | 0.96051 | 0.99659 | 0.99969 | 0.98400 | 0.98400 |
| unconstrained diffusion re-rendering 200 | 200 | 0.13360 | 0.23038 | 0.13360 | 1.00000 | 0.50578 | 0.00000 | 0.00000 |

## Main Conclusion

Pure EditGuard-style localization/copyright extraction is strong for local tamper and localized-composited AIGC edits, but collapses under unconstrained diffusion re-rendering. SCE-LocGuard adds semantic capsule interpretation, report quality calibration, and report-level aggregation on top of this dual-watermark backbone.
