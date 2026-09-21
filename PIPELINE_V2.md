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


## 0019 atomic source/final invariant

The V2 rendering path now treats erase + typeset as one commit.

- TextBubble owns source-text pixels and erase review.
- The already-associated parent Bubble owns layout/font fitting.
- Layout is validated before erase is attempted.
- Only units whose erase review passes are copied into the committed cleaned image.
- Units that fail binding, layout, mask generation, or residual review keep the original pixels and receive no Korean overlay.
- The commit audit must satisfy `EraseCommittedUnits == TypesetCommittedUnits`.
- `v2_05_committed_cleaned.webp` is built from the original image plus only committed TextBubble edits, so a failed unit cannot leave an empty balloon.
- `v2_commit_audit.json` records every requested unit as either `translated` or `original_preserved:...`.

OCR line consolidation also prefers a more complete overlapping observation when confidence is comparable, preventing a high-confidence fragment such as `THE MIDDLE` from replacing `YOU'RE STILL IN THE MIDDLE`.


## 0020 free-text / colored-glyph / four-stage audit

0019 output review exposed three remaining structural gaps:

1. RT-DETR `TextFree` detections were visible in region analysis but excluded from the V2 immutable target snapshot.
2. colored comic lettering such as red `NONONO` could survive while grayscale residual review incorrectly reported the target as clean.
3. the final audit only showed `ORIGINAL | FINAL | DIFF`, which hid whether a failure happened during erase or during commit/typesetting.

0020 changes:

- `V2DetectionSnapshot` now carries both `TextBubble` and `TextFree`.
- `TextBubble` keeps the original parent-Bubble layout rule.
- `TextFree` has no invented parent Bubble; its immutable detector rectangle is the erase target and conservative local layout anchor.
- colored glyph rescue adds filtered chroma-contrast components without OR-ing both grayscale polarities or allowing full-box erase.
- residual review is adaptive: tiny post-inpaint speckles are tolerated by density while substantial remaining lettering is still rejected.
- observed 0019 residual cases are locked as regression checks.
- completion audit is now four-stage when V2 committed-cleaned output exists:
  `1 ORIGINAL | 2 ERASE COMMIT | 3 FINAL | 4 DIFF`.
- audit summary now records V2 preservation reasons (erase review, unbound, duplicate suppression, other) and sorts pages with preserved originals first.

## 0021 regression fix

0020's colored-glyph rescue improved initial erase coverage, but reusing the same chroma detector on the inpainted image created false residuals: Telea color variation could be segmented as if source lettering survived. 0021 keeps chroma rescue for the initial erase mask, but residual review deliberately disables it and checks only normal high-contrast glyph structure near the original mask. This preserves the 0020 colored-text improvement without rejecting ordinary dialogue that 0019 already handled.

Short, wide caption/sign boxes also receive a narrowly scoped font-scale adjustment so confirmed parent-Bubble geometry is used more naturally instead of rendering a tiny label inside a wide caption. Dense narration keeps the previous conservative scale.

Free-standing, short, uppercase/emphatic dialogue without a detected parent Bubble is now treated as artwork-style graphic lettering and preserved (`stylized_graphic`) rather than erased and re-typeset with a plain font. This covers cases such as BRUCE!, NONONO and LET ME OUTTT! while leaving normal Bubble dialogue and TextFree captions on the translation path. Unbound single-character OCR fragments are reported separately as `ocr_noise` so they no longer inflate the real missed-dialogue count.


## 0022 targeted tuning

0022 narrowed the remaining failures without changing the stable core speech-bubble path.

- residual review no longer schedules a retry when the first pass is already within the accepted density range.
- flat TextFree captions on genuinely low-variance backgrounds may replace only the approved glyph mask with the local panel color, avoiding Telea edge artifacts on black narration boxes.
- the stylized_graphic heuristic was narrowed so short artwork-style shouts such as BRUCE!, BINGO! and LET ME OUTTT! remain preserved while long uppercase dialogue stays translatable.
- short wide captions use more of their confirmed parent geometry and are vertically centered instead of hugging the top edge.

On the 23-page regression set this moved committed translations from 149/190 to 158/190 and reduced erase-review failures from 14 to 6.

## 0023 best-pass erase and translation safety

0023 treats erase retry as an optional candidate, never as an unconditional replacement.

- residual review uses a tighter 2 px halo around the immutable original glyph mask.
- after a retry, the pipeline compares first-pass and retry residual scores and selects the better result per target.
- if retry is worse, only that retry footprint is restored from the first-pass cleaned image before commit.
- v2_erase_audit.json now records both raw retry residual and the selected residual/pass (SelectedResidualPixels, SelectedPass).

Translation output also gains two final safety gates:

- punctuation/noise-only corrected text such as ..., ** or _ is not sent to the final translator and cannot create invented placeholder text.
- placeholder/template responses are rejected and retried once in repair mode.
- explicit English negation such as NOT, CAN'T, DON'T, WON'T and NEVER must retain an identifiable Korean negation cue; otherwise the unit is retried once. If both the final model result and Vision draft fail validation, the unit keeps the original pixels instead of committing an unsafe translation.

These changes are intentionally scoped so the already-stable ordinary speech-bubble, colored-bubble, and graphic-SFX behavior remains unchanged.


## 0024 residual-review correction

0023 full-page review showed that several remaining failures were not erase failures at all: the cleaned debug image was already visually empty, but residual review re-segmented inpaint texture and atomic commit restored the original text.

0024 therefore leaves translation and typesetting policy untouched and changes only erase verification:

- restore the 7x7 local review halo used before 0023, because the smaller 5x5 halo weakened retry coverage on a previously successful bubble.
- keep raw residual pixels for diagnostics, but distinguish an effective residual: text-like pixels inside the original erase mask are not treated as surviving source text, because those pixels were already replaced by inpaint.
- retry is driven by suspicious residual outside the immutable original glyph mask.
- raw residual remains a sanity bound, so a wildly inconsistent target is still rejected instead of being accepted solely because its outside-mask count is low.
- best-pass selection remains in place: retry can improve a target, but can never replace a cleaner first pass.
- v2_erase_audit.json records raw and effective residual counts before and after retry for direct diagnosis.

The goal is to stop visually clean balloons/captions from being restored by a false-positive residual check while preserving the atomic erase+typeset invariant.


## 0025 source-glyph persistence review

0024 regression output showed that the five remaining erase-review failures shared one mechanism: the source text was already visually erased, but nearby high-contrast structures (bubble borders, panel/artwork lines, or inpaint texture) were re-segmented as residual text and atomic commit restored the original pixels.

0025 keeps OCR, translation, layout, erase geometry, and atomic commit policy unchanged and replaces only the residual decision signal:

- the mask builder can now expose an undilated source-glyph core separately from the dilated erase mask.
- the 7x7 halo remains available for finding retry candidates, but presence anywhere in the halo is no longer sufficient to reject a target.
- residual connected components must genuinely intersect the immutable source-glyph core before they can influence review.
- the reviewer then compares the original and cleaned pixels only at those core-linked locations. A target fails only when source-like glyph pixels actually persist after erase.
- nearby bubble borders or artwork that merely pass through the halo are diagnostic noise and no longer trigger restore.
- retry is driven by core-linked residual components; best-pass selection compares source-glyph persistence and can still keep the cleaner first pass.
- audit JSON retains the 0024 raw/outside-mask metrics for diagnosis and adds core-mask size, core overlap, persistent-core counts, selected persistence ratio, and an explicit ReviewReason.

This specifically targets the observed sequence:
erase succeeds -> reviewer sees unrelated contrast -> review fails -> atomic commit restores English.

Graphic/emphasis dialogue such as NONONO remains a separate policy question and is intentionally unchanged in 0025.


## 0026 primary-gate / secondary-rescue review

0025 proved that source-glyph persistence can rescue some real false positives (for example small text that was already visibly erased), but using it as the primary reviewer caused broad regressions on pages that 0024 already handled correctly.

0026 therefore restores the proven 0024 residual-density review as the primary gate and makes the 0025 glyph-persistence logic secondary-only:

- if the 0024 primary review passes, the unit commits immediately and the secondary reviewer is not allowed to overturn it.
- if the primary review still fails after the normal retry/best-pass selection, only then run the undilated glyph-core persistence check.
- a clean secondary result rescues the primary failure and commits the erase/typeset atomically.
- only targets that fail both layers keep the original pixels.
- retry geometry and best-pass selection stay on the 0024 residual signal, preventing the 0025 persistence metric from destabilizing already-good pages.
- v2_erase_audit.json now records PrimaryReview, SecondaryReview, RescuedBySecondary, and ReviewReason.
- three extra debug overlays are written for primary failures only:
  - v2_01b_glyph_core.webp (green)
  - v2_03a_core_linked_residual.webp (yellow)
  - v2_03b_persistent_core.webp (red)

The design goal is monotonic safety relative to 0024: existing primary-pass units cannot regress because of the experimental secondary reviewer, while known 0024 false positives can still be rescued.
