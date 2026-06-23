# Stage 8F Report-Level Region Aggregation

Stage 8F is a report-level aggregation stage.
It does not change watermark embedding, robust payload, or localization decoder.
It addresses the main Stage 8E residual problem: tiny fragmented regions cause many unknown reports.
It converts dense fragmented region reports into compact audit-level reports.
Report filtering must preserve tamper coverage and not merely delete hard cases.

## Best Strategy

- best_strategy: F3_merge_nearby_fragments
- unknown_rate_after: 0.14482758620689656
- report_reduction_ratio: 0.7049847405900305
- mean_gt_tamper_area_coverage: 0.9647786436739659
- mean_pred_area_retained: 1.0
- no_report_images: 0
- mean_reports_per_image: 5.8
- recommended_by_tradeoff: True
- anti_cheating_flag: False

No training, Stable Diffusion, SAM, CLIP, DINO, OmniGuard, payload change, or 128-bit payload is used.
