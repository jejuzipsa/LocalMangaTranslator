# LocalMangaTranslator pipeline refactor

This document fixes the intended processing order before more detection heuristics are added.

## Why this refactor exists

The 0001-0004 regression set exposed three related failure modes:

1. OCR misses an entire speech balloon, so no translation/erase work starts for it.
2. OCR mistakes artwork for text, and that false observation can eventually reach erase.
3. Vision can repair OCR too aggressively or borrow neighboring dialogue, while erase is still tied too closely to OCR boxes.

The pipeline therefore treats OCR, container geometry, semantic review, erase and layout as separate evidence stages.

## Target stages

1. Page analysis
2. Page-wide container candidate detection
3. Container candidate validation
4. Global OCR 1x
5. Global OCR 2x
6. Focus OCR
7. Container OCR passes
8. OCR observation merge
9. OCR line validation
10. Line-to-container assignment
11. Reading order
12. Translation unit creation
13. Vision OCR review
14. Translation unit validation
15. Final translation
16. Erase candidate detection
17. Erase validation
18. Layout planning
19. Layout validation
20. Inpaint (only after layout approval)
21. Typeset/render
22. Post-render validation

## Non-negotiable invariants

- OCR output is an observation, not authority.
- One physical container should normally own one translation unit.
- One canonical text line can have at most one active owner.
- Translation approval does not imply erase approval.
- Erase must have independent spatial evidence.
- Final erase pixels must remain inside an approved safe area.
- Layout failure means zero deleted pixels for that unit.
- Uncertain artwork text and uncertain SFX preserve the original.
- Container detection must eventually be able to run without an OCR seed.
- Vision may use neighboring text as context, but may not merge or duplicate neighboring units.

## Migration order

### Stage 0 - explicit OCR observations (implemented)

Preserve every OCR pass before merging:
- global 1x
- global 2x
- focus 3x
- future container 1x/2x/3x

The legacy merged OCR line list is still produced so 0004 behavior remains the baseline.

### Stage 1 - page-wide container candidates (implemented)

Add a detector that scans the page independently of OCR seeds.
Existing BalloonMaskService contour/flood-fill logic can be reused, but candidate generation and candidate validation become separate responsibilities.

### Stage 2 - container-specific OCR (implemented, conservative admission)

Geometry candidates are independently screened for valid masks, fill, score and border contact.
This is eligibility to attempt OCR, not confirmation of a balloon or permission to erase.
Every eligible candidate receives exact-bounds 1x and 2x crops even when global OCR found zero lines.
Coordinates return to page space using crop origin and scale; SourceKey retains the candidate ID.

All raw container observations are recorded. A new observation enters the legacy merged line list
only when both scales agree on normalized text and location, both confidence values are at least
0.65, and at least 90% of each box lies inside the candidate mask. Single letters, one-pass-only
findings, conflicting text and off-mask findings are held in diagnostics. These conservative
thresholds require evaluation on real pages; repeated OCR is not proof of semantic correctness.

The global OCR cache remains unchanged; container crops currently rerun on every page attempt.
Crop failures and empty results are recorded per pass. Cancellation propagates; temporary files
are removed even when a crop fails. Diagnostics add `00_container_ocr_validation.json` with
candidate eligibility, pass attempts and per-observation decisions.

Admission currently feeds the existing OCR-seeded unit builder and existing render planning.
Page candidates do not yet directly own translation units or erase masks. Full canonical line
provenance/ownership and independent erase decisions are still Stage 3/5 work.

### Stage 3 - line validation and ownership

Validate each observation/line using:
- confidence
- repeated detection across passes
- local text geometry
- container membership
- isolation from neighboring lines
- optional Vision confirmation

Only validated lines may become erase evidence.

### Stage 4 - container-scoped Vision review

Vision primarily receives one container crop and that container's OCR observations.
Page context is passed separately so adjacent dialogue cannot silently become part of the current unit.

### Stage 5 - erase decision pipeline

Create erase decisions independently from translation decisions.
A unit can be translated while one suspicious OCR line remains erase=false.

### Stage 6 - post-render validation

Detect:
- duplicated translation
- large unexpected image changes
- text remaining inside an approved container
- render outside the safe layout area

Pages with warnings stay reviewable instead of silently damaging artwork.

## Current transitional structure

The legacy pipeline remains operational while contracts are introduced.

Current:
page candidates -> OCR eligibility -> global/focus OCR + container 1x/2x -> container evidence validation -> merged OCR lines -> preliminary grouping -> OCR-seeded container ownership -> Vision -> translation -> render planning -> erase -> layout -> render

Target:
page/container candidates + OCR observations -> validated line ownership -> one translation unit per container -> Vision -> translation -> erase plan + approved layout -> independently validated erase -> render -> post validation


## Verification for the container OCR continuation

`dotnet run --project tests/PipelineChecks/PipelineChecks.csproj -c Release` checks candidate
geometry, mask membership in page coordinates, multi-scale agreement, conflicting/neighboring
text, weak observations and cancellation without invoking OCR or Ollama. Quick Build Check
runs these checks on Windows. Actual OCR recognition quality, performance and artwork
preservation still need a 23-page comparison against 0004/0005 on the user's Windows setup.
