LocalMangaTranslator test pic 구조

current
- 현재 확인 중인 최신 테스트 결과.

regression
- 최근 회귀 문제를 다시 확인하기 위한 집중 테스트.

chapter
- 최근 챕터 단위 전체 테스트.

현재 유지 세트
- current/0047
- regression/0046
- chapter/0046_2

정리 원칙
- 오래된 테스트 결과는 현재 트리에서 삭제한다.
- 필요 시 Git 히스토리의 과거 커밋에서 다시 확인할 수 있다.
- 새 테스트는 우선 current 아래에 추가하고, 검증 후 필요한 대표 결과만 남긴다.
