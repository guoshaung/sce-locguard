# Stage 10A LC Scale1000 Note

## Constraints

- no training
- no VLM
- no 128-bit payload
- no watermark embedding modification
- robust payload remains 64-bit
- verification does not access original image
- GT/edit mask is used only for evaluation

## Generation

- generated: 1000 / 1000
- success rate: 1.0
- runtime seconds: 1410.0783812999725
- runtime seconds/image: 1.0033553121089935
- peak GPU memory MB: 11575.60595703125
- by attack type: {'object_removal': 250, 'inpainting': 250, 'local_replacement': 250, 'local_style_edit': 250}

## Localized Composite

- constructed: 1000 / 1000
- outside-mask preserved: 1000
- inside-mask changed: 1000

## Evaluation

- completed: True
- failed_stage: None
- IoU: 0.9573244668616256
- Dice: 0.9778285735841605
- Precision: 0.9605053632820596
- Recall: 0.9965887139596729
- bit accuracy: 0.9996875
- payload recovery: 0.984
- capsule recovery: 0.9997
- auth success: 0.984
- Stage8E unknown: 0.7790603716957174 -> 0.44226403478394705
- F3 reports: 5933
- F3 reports/image: 5.933
- F3 unknown: 0.23141749536490813
- GT coverage: 0.9965887139596729
- predicted area retained: 1.0
- no-report images: 0
- F4_top3 reports/image: 2.975
- F4_top3 GT coverage: 0.9965861568500809
- F4_top5 reports/image: 4.813
- F4_top5 GT coverage: 0.9965864105109133

## Unknown Diagnosis

- Stage8E unknown after: 0.44226403478394705
- F3 unknown: 0.23141749536490813
- interpretation: LC localization and payload recovery remain strong. Remaining unknown labels should be treated mainly as verification-side semantic readout limitations under AIGC edit appearance changes, not as localization failures.

## Recommendation

LC scale1000 should be used as the fair local AIGC editing evaluation. Unconstrained diffusion remains a separate watermark-laundering stress test.
