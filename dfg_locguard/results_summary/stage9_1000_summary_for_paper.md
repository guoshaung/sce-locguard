# Stage 9C 1000-Sample Full Evaluation Summary

## Constraints

- No training.
- No VLM.
- No 128-bit payload expansion.
- No watermark embedding modification.
- Robust payload remains 64-bit.
- Verification does not access original images.
- GT masks are used only for evaluation.

## Basic Status

- completed: `True`
- failed_stage: `None`
- runtime: `2050s` (34.17 min)
- runtime/sample: `2.050s`
- output_dir: `dfg_locguard/outputs/stage9_full_evaluation_1000`

## Localization

- Stage1 generated: `1000`
- Stage8D regions: `33895`
- Stage8D valid regions: `33438`
- IoU: `0.892833`
- Dice: `0.939821`

## Robust Payload / Capsule / Auth

- bit accuracy: `0.999719`
- payload recovery: `0.989000`
- capsule recovery: `0.999650`
- auth success: `0.989000`
- payload failed samples: `11`
- auth failed samples: `11`
- capsule failed samples: `8`

## Stage 8E Unknown Reduction

- unknown before: `0.568432`
- unknown after: `0.307745`
- absolute reduction: `0.260687`
- relative reduction: `0.458608`
- change type distribution after Stage8E: `{'color_change': 14919, 'unknown': 10431, 'texture_change': 7470, 'structure_change': 1075}`

## Stage 8F Aggregation

- best strategy: `F3_merge_nearby_fragments`
- F3 reports: `5783`
- F3 reports/image: `5.783`
- F3 unknown rate: `0.136434`
- F3 GT coverage retained: `0.964517`
- F3 predicted area retained: `1.000000`
- F3 no-report images: `0`
- F4 top3 unknown/report count: `0.06836461126005362` / `2984`
- F4 top5 unknown/report count: `0.09380710659898477` / `4925`

## Success Criteria

- completed_true: `PASS`
- failed_zero_or_rare: `PASS`
- f3_unknown_lt_0p18: `PASS`
- f3_reports_per_image_around_5_to_7: `PASS`
- predicted_area_retained_gt_0p95: `PASS`
- gt_coverage_retained_gt_0p90: `PASS`
- no_report_images_near_zero: `PASS`
- payload_recovery_gt_0p95: `PASS`
- auth_success_gt_0p95: `PASS`
- capsule_recovery_gt_0p99: `PASS`

Overall: `PASS`

## Failure Analysis

- payload/auth related failed sample ids: `['0070', '0106', '0116', '0327', '0499', '0517', '0650', '0673', '0681', '0741', '0759']`
- diagnostic flags: `['failed samples have higher fragmentation than typical successful samples', 'failed samples have higher predicted region count than typical successful samples']`

## Recommended Next Step

The 1000-sample run passes the predefined criteria. The next stage should evaluate a real AIGC editing attack subset and add baseline comparison.
