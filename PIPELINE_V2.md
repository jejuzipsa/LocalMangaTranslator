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


## 0027 cleaned checkpoint audit

0027 does not change erase, OCR, translation, layout, primary review, or secondary rescue policy. It adds the missing structural checkpoint between erase and typesetting so the pipeline can explicitly answer: did every detector-owned erase target produce a verified empty result before Korean text is placed?

The checkpoint is grouped by immutable detector identity rather than by OCR fragments:

- TextBubble targets that share the same parent BubbleRegionId are counted as one Bubble checkpoint.
- TextFree targets are counted independently.
- an unparented TextBubble is retained as a separate diagnostic item instead of being silently dropped.
- a Bubble checkpoint is EMPTY_OK only when every selected TextRegionId owned by that Bubble has a clean erase audit.
- verification uses exact ID-set equality, not count equality alone, so replacing one missing Bubble with another cannot accidentally pass.

New debug output:

- v2_04b_cleaned_checkpoint.webp
  - thin gray boxes: RT-DETR Bubble detections that are not erase targets.
  - green boxes: erase-target Bubble/TextFree checkpoints verified EMPTY_OK.
  - red boxes: erase-target checkpoints still marked ERASE_CHECK.
  - the top summary shows verified/target counts separately for Bubble and TextFree.

v2_erase_audit.json now also records:

- DetectedBubbleCount
- EraseTargetBubbleCount
- EmptyVerifiedBubbleCount
- EraseTargetBubbleIds
- EmptyVerifiedBubbleIds
- CheckpointMissingBubbleIds
- EraseTargetTextFreeCount
- EmptyVerifiedTextFreeCount
- the corresponding TextFree ID sets
- per-checkpoint bounds, owned TextRegionIds, EMPTY_OK/ERASE_CHECK status, and OverallPass

This is diagnostic-only in 0027. A checkpoint result does not yet override the 0026 restore/commit behavior. The purpose is to verify the cleaned stage independently and determine whether later restore logic is undoing a visually complete erase.


## 0028 independent cleaned-state verification

0028 separates the question "did the pixel reviewer like the erase?" from the question "does the cleaned image still contain detector-visible text?".

The 0026 residual-density + glyph-persistence reviewer remains unchanged as diagnostic evidence. It no longer owns the normal V2 commit decision when the independent cleaned verifier is available.

New flow:

1. freeze the original RT-DETR Bubble/TextBubble/TextFree IDs and geometry.
2. erase only the selected immutable text targets exactly as before.
3. save v2_04_cleaned_final.webp.
4. run RT-DETR again on that cleaned image.
5. compare only cleaned TextBubble/TextFree detections against the original selected TextRegion bounds.
6. if no cleaned text detection matches an original target, that target is EMPTY_OK.
7. a translation unit is committed only when every immutable TextRegionId it owns is EMPTY_OK.
8. the old pixel reviewer result is retained beside the semantic result so false-positive reviewer failures are visible.

The second RT-DETR pass is evidence only. It never replaces the original target IDs, never moves the erase geometry, and never creates a new translation/layout target.

Debug separation:

- v2_04a_legacy_reviewer_checkpoint.webp
  - the 0027-style visualization driven by the old pixel reviewer.
- v2_04b_cleaned_checkpoint.webp
  - the independent cleaned-image RT-DETR verification.
  - green: EMPTY_OK.
  - red: TEXT_REDETECTED.
  - magenta: text regions detected on the cleaned image.
- v2_cleaned_state_audit.json
  - original target IDs, cleaned detections, target-level matches, Bubble/TextFree checkpoint counts, ID-set equality, and legacy-reviewer disagreements.

Commit audit schema is now pipeline-v2-cleaned-state-commit-v2. It records both LegacyEraseReviewClean and CleanedStateEmpty. If the cleaned verifier is unavailable unexpectedly, the pipeline falls back to the legacy reviewer rather than weakening safety.

0028 intentionally does not yet make the whole cleaned raster immutable for every downstream failure and does not yet add the erase-only UI mode. Those are the next structural steps after the 23-page regression run confirms that semantic cleaned verification fixes the two known false restores without creating new misses.


## 0029 glyph-topology-preserving erase masks

0028 fixed false source restoration by separating cleaned-state verification from the legacy pixel reviewer. The 23-page regression then exposed a different failure class in stylized American-comic lettering: RT-DETR could correctly identify the TextBubble while the pixel mask itself damaged the wrong pixels.

The root cause was the mask component filter using an external contour and filling that contour solid. For outlined glyphs such as O/P/R, a solid contour fill converts the glyph's white counter into an erase pixel. Closely spaced display letters can suffer the same problem in their negative space. Telea then pulls nearby black outline pixels into those incorrectly masked white areas, producing black fills/blotches even though detector geometry was correct.

0029 changes mask construction rather than reviewer thresholds:

- accepted contours are still used to reject large/background components, but the output copies only foreground pixels that actually existed in the threshold mask.
- glyph holes and inter-letter negative space are therefore preserved.
- color rescue now keeps its real chroma seed pixels and adds only nearby high-contrast outline pixels, bounded to a small radius around the accepted color seed.
- the normal final 3x3 erase dilation remains, but it now expands a topology-correct foreground mask instead of an already flood-filled silhouette.
- RT-DETR geometry and the 0028 independent cleaned-state verifier remain unchanged.

New debug evidence:

- v2_01a_grayscale_mask.webp: grayscale/Otsu erase contribution.
- v2_01c_color_rescue_mask.webp: chroma + local outline rescue contribution.
- v2_01_text_mask.webp: final merged erase mask.

v2_erase_audit.json now records aggregate and per-target GrayscaleMaskPixels and ColorRescueMaskPixels. This makes cases such as a color-only stylized target explicit instead of hiding them behind InitialMaskPixels.

0029 is intended as a general American-comic display-lettering fix, not a word-specific NONONO exception.


## 0030 background reconstruction strategy

0029 showed that a correct foreground glyph mask is not enough. Large colored display lettering could be segmented correctly while Telea still reconstructed the erased pixels from nearby red/black ink, producing colored blotches. The problem is therefore split explicitly into two stages:

1. foreground segmentation: which pixels belong to lettering?
2. background reconstruction: what should replace those pixels?

0030 keeps the 0029 topology-preserving mask and the 0028 independent cleaned-state verifier. It adds a conservative reconstruction router per immutable TextRegionId.

### FLAT_FILL

For a target whose visible non-glyph pixels inside the original detector text box form one dominant local color cluster, the target is classified as FLAT_FILL. The background color is estimated robustly from non-mask pixels inside the text box rather than from an outside ring. This matters for large display lettering whose TextBubble nearly fills the parent Bubble.

The accepted glyph mask is then replaced directly with that estimated background color. This prevents Telea from pulling colored glyph fill or black outline pixels back into a flat speech balloon or caption panel.

### TELEA

If the local background is not confidently flat, the target stays on the existing Telea path. Artwork, gradients, textured panels, and ambiguous targets therefore do not receive a solid-color fill.

The initial reconstruction pass may contain both strategies on one page. Flat targets are filled first; only masks assigned to TELEA are passed to inpainting.

### Flat classifier evidence

For each target the audit records:

- Strategy: FLAT_FILL or TELEA
- mask pixel count
- non-glyph background sample count
- estimated B/G/R background color
- dominant color match ratio
- 75th and 90th percentile color distance
- decision reason
- post-reconstruction masked-pixel background distance

The flat decision is intentionally conservative. Ambiguous backgrounds fall back to TELEA.

### New debug outputs

- v2_02a_reconstruction_strategy.webp
  - green: FLAT_FILL
  - orange: TELEA
  - header counts each strategy and flat-background quality failures.
- v2_background_audit.json
  - per-target reconstruction evidence and post-reconstruction quality result.

The same information is also embedded in v2_erase_audit.json under BackgroundReconstruction.

0030 does not special-case NONONO or any literal word. The intended regression is general: colored/outlined lettering on a flat balloon should restore the balloon background, while text over real artwork should remain on the inpainting path.
