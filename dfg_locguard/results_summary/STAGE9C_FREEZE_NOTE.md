# Stage 9C Freeze Note

## Frozen Code State

- branch: `stage9-server`
- commit hash: `e49e1b4f3f1e4b4396e935627580142d8e8418ea`
- output directory: `/data/watermark_exps/dfg_locguard/outputs/stage9_full_evaluation_1000`
- paper results zip: `/data/watermark_exps/stage9_1000_paper_results.zip`

## Constraints

- No training.
- No VLM.
- No 128-bit payload expansion.
- No watermark embedding modification.
- Robust payload remains 64-bit.
- Verification does not access original images.
- GT masks are used only for evaluation.

## Runtime

- total runtime: `2050s` (34.17 min)
- runtime/sample: `2.050s`

## 1000-Sample Key Metrics

- completed: `True`
- failed_stage: `None`
- Stage1 generated: `1000`
- Stage8D regions: `33895`
- Stage8D valid regions: `33438`
- IoU: `0.892833`
- Dice: `0.939821`
- bit accuracy: `0.999719`
- payload recovery: `0.989000`
- capsule recovery: `0.999650`
- auth success: `0.989000`
- Stage8E unknown: `0.568432 -> 0.307745`
- Stage8E absolute reduction: `0.260687`
- Stage8E relative reduction: `0.458608`

## Stage 8F Main Strategy

- best strategy: `F3_merge_nearby_fragments`
- F3 reports: `5783`
- F3 reports/image: `5.783`
- F3 unknown rate: `0.136434`
- F3 GT coverage retained: `0.964517`
- F3 predicted area retained: `1.000000`
- F3 no-report images: `0`

## Compact Ablations

- F4_top3 reports: `2984`
- F4_top3 reports/image: `2.984`
- F4_top3 unknown rate: `0.068365`
- F4_top3 GT coverage: `0.962572`
- F4_top3 predicted retained: `0.993700`
- F4_top5 reports: `4925`
- F4_top5 reports/image: `4.925`
- F4_top5 unknown rate: `0.093807`
- F4_top5 GT coverage: `0.963521`
- F4_top5 predicted retained: `0.996093`

## Payload/Auth Failures

- payload failed samples: `11`
- auth failed samples: `11`
- capsule failed samples: `8`
- bit failed samples: `11`
- failed sample IDs: `0070, 0106, 0116, 0327, 0499, 0517, 0650, 0673, 0681, 0741, 0759`
- diagnostic flags: `['failed samples have higher fragmentation than typical successful samples', 'failed samples have higher predicted region count than typical successful samples']`

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

- all success criteria passed: `True`

## Next Step

Use this frozen 1000-sample result as the main stability experiment. Next recommended experiment: real AIGC editing attack subset and baseline comparison.
