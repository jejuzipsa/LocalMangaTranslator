# Architecture

## 사용자 흐름

이미지/폴더 추가
→ OCR
→ Vision LLM: OCR 검수 + 한국어 번역
→ 인페인트
→ 한글 조판
→ 결과 저장

## 원칙

- StoneCandy 코드에 의존하지 않는 새 구현
- UI는 VOSR처럼 단순하게 유지
- 모델은 폴더의 model.json 프로필로 교체 가능
- 대형 모델 바이너리는 Git에 포함하지 않음
- 모델 다운로드/처리 상태는 하단 작업 로그에 표시
- 일반 커밋은 빠른 컴파일 검사만 수행
- 포터블 Windows 빌드는 필요할 때 수동 실행

## 구현됨

- WPF 다크 UI
- 이미지/폴더 드래그
- 파일/폴더 추가, 전체 삭제
- 저장 폴더 선택/열기
- 작업 큐, 진행률, 중지
- 작업 로그
- 모델 manifest 자동 탐색
- RapidOCR/PP-OCRv5 다중 패스 OCR
- Ollama 서버/모델 설치 상태 확인
- Ollama 모델 다운로드 진행 로그
- Qwen Vision에 원본 이미지 + OCR 결과 전달
- Vision LLM의 OCR 교정 + 한국어 번역 JSON 파싱
- 페이지별 translation.json 저장
- 커밋용 빠른 Windows 빌드 Action
- 수동 포터블 빌드 Action

## 다음

1. OCR/번역 JSON 실제 샘플 검증
2. 인페인트 모듈 연결
3. 말풍선/텍스트 영역 기준 한글 조판
4. 완성 이미지 저장
