# Pipeline V2 rewrite

This branch is a clean rewrite of the image-processing path while keeping the existing WPF UI, model selection, queue, output layout, translation models, and final delivery shell.

## Non-negotiable rule

RT-DETR Bubble/TextBubble geometry is immutable evidence.

Downstream OCR grouping, confidence ranking, Vision correction, translation, ownership, or canonical-line deduplication may not shrink, replace, or delete the geometry used by the erase pipeline.

## Stage order

1. Detection
   - RT-DETR Bubble/TextBubble
   - create immutable `V2DetectionSnapshot`

2. Text mask / erase
   - operate directly on TextBubble crops
   - no dependency on `OcrContainerUnitBuilder`, `DeduplicateLines`, `BalloonMaskService`, `no_safe_text_line`, or translation text
   - current V2 probe uses an Apache-2.0 Comic Translate-style black/white component mask and OpenCV Telea only as the first measurable baseline

3. OCR / translation
   - remains separate from erase geometry
   - OCR may read and correct text, but it cannot redefine V2 erase targets

4. Typesetting
   - existing layout work can be reused after V2 erase is validated

5. Audit
   - compare the same page across:
     - V2 detection
     - V2 text mask
     - V2 cleaned image
     - final render

## External components: exact status

| Component | Role | Status in this branch |
| --- | --- | --- |
| ogkalu2 Comic Translate RT-DETR-v2 | Bubble/TextBubble detection | integrated |
| Comic Translate content-mask approach | TextBubble-local pixel mask | integrated as C# adaptation |
| RapidOCR / PP-OCR | OCR | existing, kept separate from erase |
| Baberu OCR | secondary OCR | existing, kept separate from erase |
| comic-text-detector (CTD) | dedicated learned text segmentation | candidate for the next mask comparison |
| LaMa / manga LaMa | learned inpainting | candidate after mask quality is proven |
| AOT-GAN | alternate inpainting | candidate/fallback experiment |

A component is not called "integrated" until executable code uses it in the V2 path.

## First milestone

The first V2 milestone intentionally does not replace the final renderer yet. It writes independent debug outputs from RT-DETR before OCR/translation can mutate anything:

- `debug/<page>.v2_00_detection.webp`
- `debug/<page>.v2_01_text_mask.webp`
- `debug/<page>.v2_02_cleaned.webp`
- `debug/<page>.v2_detection.json`

This lets the 22-page regression set answer one question first: **can we reliably empty the detected text regions without losing text that RT-DETR already found?**

Only after that is stable will V2 cleaned output become the input to typesetting.
