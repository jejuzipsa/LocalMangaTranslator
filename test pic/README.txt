LocalMangaTranslator test pic 구조

current
- 현재 확인 중인 최신 테스트 결과.

regression
- 특정 회귀 문제를 다시 확인하기 위한 집중 테스트.

chapter
- 여러 페이지 또는 챕터 단위 전체 테스트.

archive
- 이전 개발 단계의 테스트 결과.
- 비교/추적용으로 보존하며 현재 검수에서는 우선순위가 낮음.

정리 원칙
- 테스트 결과는 삭제하지 않고 Git 히스토리와 archive에 보존한다.
- 최신 검증이 끝나면 current 결과는 필요에 따라 regression/chapter/archive로 이동한다.
- 새 테스트는 우선 current 아래에 버전 폴더로 추가한다.
