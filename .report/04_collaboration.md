# 4. 협업 기록 및 참고 문헌

## 4.1 팀원별 역할 분담 및 협업 이력

### 역할 분담

| 학번 | 이름 | 담당 역할 |
|------|------|-----------|
| 3201 | 공덕현 | 주제 선정, 데이터 수집, 보고서 작성 |
| 3207 | 박동현 | 주제 선정, 모델 설계·학습·평가 |
| 3212 | 조재민 | 주제 선정, 데이터 수집, 보고서 작성 |

## 4.2 생성형 AI 활용 출처

| 활용 목적 | 모델 | 반영 내용 |
|-----------|------|-----------|
| 프로젝트 구조 설계 및 코드 리뷰 | Claude | EfficientNetEmbedder 구조, ArcFaceLoss 구현, AnimeRetriever FAISS 래핑 설계 보조 |
| 학습 스크립트 디버깅 | Claude | MPS autocast 설정, 2-Phase 학습 루프 최적화 |
| EDA 및 전처리 코드 작성 | Claude | `notebooks/eda.ipynb`, `src/datasets/preprocess.py` 작성 |
| 모델 비교 노트북 작성 | Claude | `notebooks/compare_models.ipynb`, `scripts/compare_models.py` 작성 |
| 소스 파일 docstring 추가 | Copilot | 전체 소스 파일 docstring 자동 생성 |
| Retrieval 데모·평가 스크립트 작성 | Claude | `scripts/demo.py` 신규 (query/gallery 분리, FAISS top-k 그리드 시각화), `scripts/eval.py` self-match 오염 제거 및 체크포인트 형식 호환 수정 |

## 4.3 참고 문헌 및 자료 출처

1. Zheng, H., et al. "AniWho: A Quick and Accurate Way to Classify Anime Character Faces in Images." arXiv:2208.11012 (2022).
2. Yashs744. Anime-Character-Detector. https://github.com/Yashs744/Anime-Character-Detector
3. Tarun Dalal. Top 15 Anime Main Characters Dataset. https://www.kaggle.com/datasets/tarundalal/top-15-anime-main-charcters
