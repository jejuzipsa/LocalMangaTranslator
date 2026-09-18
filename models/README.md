# Models

모델 바이너리는 Git 저장소에 넣지 않습니다.

## Vision LLM

`models/vision_llm/<id>/model.json`을 추가하면 프로그램 시작 시 자동으로 목록을 읽습니다.

초기 프로필:
- Qwen 3.5 9B
- provider: Ollama
- tag: `qwen3.5:9b`

향후 Qwen 신버전, GLM 또는 다른 Vision LLM을 같은 방식으로 추가할 수 있습니다.

## OCR

현재 OCR 엔진은 PP-OCRv5/RapidOCR 계열입니다.

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

아직 구현하지 않았습니다. Vision LLM 단계 다음에 독립 모듈로 연결합니다.
