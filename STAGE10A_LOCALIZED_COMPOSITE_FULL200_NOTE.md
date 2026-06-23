# Stage 10A Localized-Composite Full 200 Note

## Constraints

- no training
- no VLM
- no 128-bit payload
- no watermark embedding modification
- robust payload remains 64-bit
- verification does not access original image
- GT/edit mask is used only for evaluation

## Construction

- completed cases: 200 / 200
- failed construction: 0
- outside-mask preserved pass count: 200
- inside-mask changed pass count: 200
- cases by attack type: {'object_removal': 50, 'inpainting': 50, 'local_replacement': 50, 'local_style_edit': 50}

## Evaluation

- completed: True
- failed_stage: None
- eval runtime seconds: 227.72882533073425
- IoU: 0.9546367875095549
- Dice: 0.9758166158828507
- Precision: 0.9578357058095728
- Recall: 0.9965713142785205
- bit accuracy: 0.999765625
- payload recovery: 0.99
- capsule recovery: 1.0
- auth success: 0.99
- F3 unknown: 0.22920203735144312
- F3 reports/image: 5.89
- GT coverage retained: 0.9965713142785205
- predicted area retained: 1.0
- no-report images: 0
- F4_top3 mean_reports_per_image: 2.97
- F4_top3 GT coverage: 0.9965675019826142
- F4_top5 mean_reports_per_image: 4.78
- F4_top5 GT coverage: 0.99656825719176

## Interpretation

Localized-composite full 200 preserves the unmasked watermarked region and should be treated as the fair local AIGC editing evaluation. The original Stage10A unconstrained diffusion result should be retained as a global diffusion re-rendering / watermark laundering stress test.
