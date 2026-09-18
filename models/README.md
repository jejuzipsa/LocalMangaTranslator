# Models

모델 바이너리는 Git 저장소에 넣지 않습니다.

## Vision / Translation LLM

`models/vision_llm/<id>/model.json`을 추가하면 프로그램 시작 시 자동으로 목록을 읽습니다.

현재 프로필:
- Gemma 4 12B
  - tag: `gemma4:12b`
  - tasks: `review`, `translation`
- Qwen 3.5 9B
  - tag: `qwen3.5:9b`
  - tasks: `review`, `translation`
- Qwen 3.8 27B HQ
  - tag: `qwen3.8:27b`
  - tasks: `review`, `translation`
- Aya Expanse 8B
  - tag: `aya-expanse:8b`
  - tasks: `translation`
  - text-only translation model; not shown in OCR review model list

예시:

```json
{
  "id": "my-model",
  "name": "My Model",
  "type": "vision_llm",
  "provider": "ollama",
  "modelTag": "my-model:tag",
  "apiBase": "http://127.0.0.1:11434",
  "supportsImage": true,
  "supportsText": true,
  "enabled": true,
  "tasks": ["review", "translation"]
}
```

`review` 모델은 이미지 입력을 지원해야 합니다. `translation` 모델은 텍스트 입력만 지원해도 됩니다.

## OCR

OCR은 LLM과 독립된 엔진으로 동작합니다. 현재 PP-OCRv5/RapidOCR 계열입니다.

선호 경로:

```text
models/ocr/v5/
  ch_PP-OCRv5_mobile_det.onnx
  ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx
  ch_PP-OCRv5_rec_mobile.onnx
  ppocrv5_ch_dict.txt
```

파일이 없으면 RapidOcrNet 기본 모델 초기화를 시도합니다.

## Inpaint

현재 렌더러의 인페인트/조판 로직은 독립 모듈입니다. 향후 실제 인페인트 모델을 연결할 때도 `models/inpaint/<id>/model.json` 구조로 분리할 예정입니다.
