# Stage 8E Report Quality Audit

Stage 8D already validates the dual-watermark pipeline.
Stage 8E focuses on report quality, not new watermark embedding.
The goal is to reduce unknown change type rate.
Verification still does not access original image.
The current semantic explanation remains coarse-grained unless a VLM plug-in is enabled.

## Results

- total_region_reports: 983
- valid_report_count: 983
- original_unknown_rate: 0.43845371312309256
- improved_unknown_rate: 0.23397761953204477
- absolute_reduction: 0.2044760935910478
- relative_reduction: 0.46635730858468677
- auth_valid_rate: 1.0
- capsule_valid_rate: 1.0
- copyright_valid_rate: 1.0

No training, Stable Diffusion, SAM, CLIP, DINO, OmniGuard, 128-bit payload, or new watermark branch is used.
