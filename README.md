# LocalMangaTranslator

Windows용 로컬 만화 이미지 번역기.

이 저장소는 기존 프로젝트를 정리하고 처음부터 다시 설계한 새 구현입니다.

## 기본 파이프라인

Architecture 2.0부터 페이지 구조 분석과 문자 인식을 서로 독립적으로 수행합니다.

```text
이미지
├─ Page Analysis
│  ├─ 만화 전용 RT-DETR: 말풍선 / 말풍선 안 글자 / 말풍선 밖 글자
│  └─ 기존 기하 검출: 컨테이너 내부 mask 보조
│
└─ Text Analysis
   ├─ PP-OCRv5 / RapidOCR 전체 페이지 OCR
   └─ RT-DETR text region 대상 1x / 2x OCR
          ↓
      Evidence Fusion
          ↓
      Vision OCR 검수
          ↓
       최종 번역
          ↓
   안전 ErasePlan / 조판
          ↓
   WebP Lossless 결과
```

RT-DETR은 문자열을 읽지 않습니다. 페이지의 구조/텍스트 영역을 찾는 별도 detector이며,
OCR 결과와 공간 evidence를 합친 뒤에만 번역 unit을 만듭니다.

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
