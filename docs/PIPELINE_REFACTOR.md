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
18. Inpaint
19. Layout planning
20. Layout validation
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

### Stage 0 - explicit OCR observations (current)

Preserve every OCR pass before merging:
- global 1x
- global 2x
- focus 3x
- future container 1x/2x/3x

The legacy merged OCR line list is still produced so 0004 behavior remains the baseline.

### Stage 1 - page-wide container candidates

Add a detector that scans the page independently of OCR seeds.
Existing BalloonMaskService contour/flood-fill logic can be reused, but candidate generation and candidate validation become separate responsibilities.

### Stage 2 - container-specific OCR

Every approved container receives dedicated OCR crops.
Global and container observations are merged only after provenance has been retained.

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
OCR observations -> merged OCR lines -> preliminary grouping -> OCR-seeded container ownership -> Vision -> translation -> render planning -> erase -> layout -> render

Target:
page/container candidates + OCR observations -> validated line ownership -> one translation unit per container -> Vision -> translation -> independently validated erase -> layout -> render -> post validation
