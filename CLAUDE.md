# Project: 캐릭터보고 애니 찾아주기

애니 캐릭터 이미지 → 작품명 검색 시스템. Classification이 아닌 **Retrieval** 구조.

## 핵심 아키텍처

- **Backbone**: EfficientNet-B7 (AniWho arXiv:2208.11012 — top-1 85.08%)
- **Loss**: ArcFace (additive angular margin) — metric learning
- **Index**: FAISS IndexFlatIP (cosine similarity on L2-normed embeddings)
- **Embedding dim**: 512

분류 헤드 없음. 임베딩 → FAISS 검색으로 새로운 캐릭터도 인덱스 재빌드 없이 추가 가능.

## 주요 파일

| 파일 | 역할 |
|------|------|
| `src/models/efficientnet.py` | EfficientNetEmbedder (backbone + projector) |
| `src/losses/arcface.py` | ArcFaceLoss |
| `src/inference/retrieval.py` | AnimeRetriever (FAISS wrapping) |
| `src/datasets/anime_dataset.py` | AnimeCharacterDataset (ImageFolder-style) |
| `configs/train.yaml` | 학습 하이퍼파라미터 (Hydra) |
| `scripts/train.py` | 학습 진입점 |
| `scripts/export.py` | FAISS 인덱스 빌드 및 저장 |

## 데이터 구조

`data/processed/` 아래 `<캐릭터명>/` 폴더별로 이미지 정리. `AnimeCharacterDataset`이 이 구조를 기대함.

캐릭터명과 작품명 매핑은 별도 JSON 또는 폴더명 컨벤션으로 관리 (미구현 — 추후 추가).

## 개발 원칙

- 설정은 Hydra (`configs/`) 통해서만. 하드코딩 금지.
- 모델 변경 시 `configs/model.yaml`의 `backbone` 키만 수정 (timm 모델명).
- 새 캐릭터 추가 = `data/processed/`에 폴더 추가 후 `scripts/export.py` 재실행.
- `src/inference/retrieval.py`의 `AnimeRetriever`는 모델과 분리 유지. 모델 교체 시 retriever 코드 건드리지 않도록.

## 주의사항

- FAISS index는 L2 norm된 벡터를 가정 (`IndexFlatIP` = cosine). `build_index()` 내부에서 `faiss.normalize_L2()` 호출함. 외부에서 중복 정규화 금지.
- ArcFace weight는 학습 시 `num_classes`를 데이터셋에서 동적으로 받음. 체크포인트 로드 시 클래스 수 불일치 주의.
- AMP (`torch.cuda.amp`) 기본 활성화. MPS(Apple Silicon)에서는 `training.amp: false` 설정 필요.
