# LocalMangaTranslator

Windows용 로컬 만화 이미지 번역기.

이 저장소는 기존 프로젝트를 정리하고 처음부터 다시 설계한 새 구현입니다.

## 기본 파이프라인

0011부터 페이지 구조와 문자 판독을 끝까지 분리합니다.

```text
이미지
├─ RT-DETR Region Map
│  ├─ bubble
│  ├─ text_bubble
│  └─ text_free
│
├─ OCR A: PP-OCRv5 / RapidOCR
│  └─ 전체 페이지 + RT-DETR 확정 bubble focused OCR
│
└─ OCR B: Baberu OCR
   └─ RT-DETR speech-bubble crop
          ↓
      Region + OCR Evidence
          ↓
      Vision OCR 검수
          ↓
       최종 번역
          ↓
   Region SafeMask 기반
   ErasePlan + Layout
          ↓
   WebP Lossless 결과 저장
          ↓
   완료 검토 로그
```

RT-DETR이 페이지 구조를 결정하고, legacy 기하 검출은 이미 확인된 말풍선의
contour/safe-mask 보조에만 사용합니다. RapidOCR 영역 재검출 결과는 진단용으로
분리하며 canonical OCR line에 직접 합치지 않습니다.

Baberu는 별도의 만화 말풍선 OCR입니다. OCR A와 OCR B가 다르면 Vision 단계가
실제 이미지를 보고 판정하며, 어느 OCR의 문자열도 그 자체로 삭제 권한을 주지 않습니다.

## 목표 UI

- 이미지/폴더 드래그 앤 드롭
- 파일 추가 / 폴더 추가 / 전체 삭제
- 저장 위치 선택 / 폴더 열기
- OCR 검수 모델과 번역 모델을 별도로 선택
- 번역 시작 / 작업 중지
- 파일별 상태와 처리 시간
- 전체 진행률
- 하단 작업 로그

## 모델 구조

특정 LLM에 종속되지 않도록 모델 프로필을 폴더 단위로 관리합니다.

```text
models/
  vision_llm/
    <model-id>/
      model.json
  ocr/
  layout/
  inpaint/
```

`model.json`의 `tasks`로 모델이 맡을 역할을 선언합니다.

- `review`: 이미지 + OCR 결과를 보고 OCR을 교정하는 Vision 단계
- `translation`: 교정된 원문을 최종 한국어로 번역하는 단계

현재 기본 선택은 두 단계 모두 `gemma4:12b`입니다. 검수와 번역에 서로 다른 모델을 선택할 수도 있습니다.

1차 개발은 Ollama의 로컬 API를 사용하고, 이후 다른 런타임도 추가할 수 있게 설계합니다.

> 모델 파일은 저장소에 포함하지 않습니다.


## 조판

렌더 단계는 말풍선/캡션의 색을 흰색으로 가정하지 않습니다. OCR 블록 주변의 경계 구조를 찾아 컨테이너를 추정하고, 글자 픽셀만 지운 뒤 원래 컨테이너를 유지합니다. 번역문은 내부 안전영역에 맞춰 자동 줄바꿈, 폰트 크기, 줄간격을 조절합니다.


## 결과 폴더 구조

출력 루트에는 최종 번역 이미지만 남깁니다.

```text
output/
  debug/
  json/
  완료검토로그/
  page001.translated.webp
  page002.translated.webp
```

- `debug/`: 단계별 디버그 이미지와 진단 JSON
- `json/`: 페이지별 최종 `*.translation.json`
- `완료검토로그/`: 최종 결과 비교/Audit 자료
- 출력 루트: 최종 번역 이미지
