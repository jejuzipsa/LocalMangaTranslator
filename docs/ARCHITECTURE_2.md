# Architecture 2.0

LocalMangaTranslator는 OCR 결과를 페이지 구조의 출발점으로 사용하지 않는다.

## 핵심 원칙

1. Page Analysis와 Text Analysis를 독립적으로 실행한다.
2. 페이지 detector는 문자열을 결정하지 않는다.
3. OCR은 말풍선 존재 여부를 결정하지 않는다.
4. 여러 evidence는 마지막 fusion 단계에서 연결한다.
5. translation 승인과 erase 승인은 별개다.
6. 외부 detector가 실패해도 legacy pipeline으로 복귀할 수 있어야 한다.

## 0010에서 실제 연결된 흐름

### A. Page Analysis branch

- 기존 `PageContainerCandidateDetector`
  - 밝기/경계 기반 컨테이너 후보
  - 내부 mask 생성
- `RtdetrPageRegionAnalyzer`
  - 외부 만화 전용 RT-DETR-v2 ONNX
  - classes: bubble / text_bubble / text_free
- `PageAnalysisService`
  - 두 결과를 독립적으로 수집
  - 같은 물리 말풍선이면 learned detector evidence로 기존 후보 score 보강
  - 기존 기하 후보가 없지만 RT-DETR bubble이 강하면 OCR용 fallback container 생성
  - fallback container는 OCR/ownership evidence일 뿐 erase 권한이 아니다

### B. Text Analysis branch

- 기존 전체 페이지 PP-OCR/RapidOCR
- RT-DETR의 `text_bubble` / `text_free` crop에 별도 1x/2x OCR
- `TextEvidenceFusionService`
  - region cross-scale agreement
  - global OCR agreement
  - strong learned-region + strong OCR
  - 위 evidence가 없으면 region OCR은 채택하지 않음

### C. Fusion / 후반부

- accepted OCR observations를 canonical line으로 merge
- container ownership / rescue
- Vision review
- translation
- RenderPlan
- ErasePlan
- layout
- WebP lossless output

## 0010에서 의도적으로 건드리지 않는 부분

RenderPipeline의 erase safety 정책은 0009 그대로 유지한다.
새 RT-DETR box 자체가 삭제 권한을 얻지는 않는다.

다음 단계에서는 PageAnalysisResult를 RenderPipeline까지 전달해
말풍선 safe interior와 text mask를 독립적인 erase evidence로 사용하는 작업을 진행한다.

## UI

기존 UI는 유지하고 다음 두 옵션만 추가한다.

- `만화 전용 페이지 분석 (RT-DETR + 기존 검출)`
- `검출 영역 OCR 보강`

둘을 끄면 이전 방식과 직접 비교할 수 있다.
