# Third-party notices

## Comic text and bubble detector

LocalMangaTranslator can download and run the ONNX model published as
`ogkalu/comic-text-and-bubble-detector`.

- Purpose: comic speech-bubble and text-region detection
- Classes used: `bubble`, `text_bubble`, `text_free`
- Model family: RT-DETR-v2
- Model repository: https://huggingface.co/ogkalu/comic-text-and-bubble-detector
- License: Apache License 2.0
- The model is downloaded on demand and is not committed to this repository.

The .NET inference wrapper follows the public ONNX input/output contract also
used by the Apache-2.0 Comic Translate project:

- https://github.com/ogkalu2/comic-translate

No GPL source code is copied into LocalMangaTranslator.
