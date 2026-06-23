# Stage 10A Attack Locality Audit

## Purpose

This audit checks whether Stage 10A was a fair local AIGC edit or an unconstrained diffusion re-rendering stress test.

## Source Audit

- attacks using watermarked source: 200
- attacks using original source: 0
- uncertain/ambiguous source count: 0
- source conclusion: watermarked_source_confirmed_by_manifest_paths_and_generator_code

## Locality Audit

- mean mask area ratio: 0.13359886169433594
- mean changed area ratio inside mask: 0.9407307072920098
- mean changed area ratio outside mask: 0.8006020027498679
- mean abs diff outside mask: 10.446121487617493
- mean outside-mask PSNR: 24.67001243515728
- globally changed samples: 200 / 200
- locality conclusion: unconstrained_diffusion_rerendering_stress_test

## Predicted Mask Audit

- mean predicted mask area ratio: 1.0
- mean GT mask area ratio: 0.13359886169433594
- near-full-image alarms: 200 / 200

## Localized Composite 40

- completed: True
- IoU: 0.9665185712973751
- Dice: 0.9829425828168702
- bit accuracy: 1.0
- payload recovery: 1.0
- auth success: 1.0
- F3 unknown: 0.13333333333333333
- F3 reports/image: 3.0

## Three-Setting Comparison

| setting | IoU | Dice | Precision | Recall | bit acc | payload | auth | F3 unknown | reports/image | pred area | GT coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage9C_synthetic_1000 | 0.8928331973424926 | 0.9398208447718827 | 0.9210072993936756 | 0.96451697277382 | 0.99971875 | 0.989 | 0.989 | 0.1364343766211309 | 5.783 |  | 0.96451697277382 |
| Stage10A_unconstrained_diffusion_200 | 0.13359886169433594 | 0.23037736446106755 | 0.13359886169433594 | 1.0 | 0.50578125 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| Stage10A_localized_composite_diagnostic_40 | 0.9665185712973751 | 0.9829425828168702 | 0.9688461451110904 | 0.9975210389299878 | 1.0 | 1.0 | 1.0 | 0.13333333333333333 | 3.0 | 0.15695590972900392 | 0.9975177266768928 |

## Interpretation

- Original Stage10A failure is mostly caused by global generative reconstruction outside the edit mask.
- recommendation_rerun_full200_with_localized_compositing: True
- recommendation_proceed_stage10b: True

## Constraints

- no training
- no VLM
- no 128-bit payload
- no watermark embedding modification
- robust payload remains 64-bit
- verification does not access original image
- GT mask is used only for evaluation
