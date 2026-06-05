# 캐릭터보고 애니 찾아주기

애니메이션 캐릭터 이미지 한 장으로 작품명을 찾아주는 딥러닝 기반 이미지 검색 시스템.

쇼츠, 인스타그램 릴스, 커뮤니티에서 발견한 애니 캐릭터 — 출처를 몰라 몇 달째 갈증을 느끼는 유저를 위해 만들었습니다.

---

## 탐구 주제 및 이론적 배경

### 1.1 주제명
캐릭터보고 애니 찾아주기

### 1.2 주제 선정 이유 및 목적

숏츠나 커뮤니티를 보다가 애니 캐릭터가 이뻐서 애니가 궁금해졌는데 못 찾은지 3달이 지났다고 함. 이 문제는 단순 개인 경험이 아닙니다.

- **작품 찾기 귀찮은 해결**: 캐릭터 이름조차 모를 때 검색 자체가 불가능
- **궁금증 해소**: 출처 모를 2차 창작/팬아트의 원작 탐색
- **팬덤 진입 장벽 낮추기**: 찾지 못해 그냥 넘어가던 잠재적 팬 포착

쇼츠, 인스타그램 릴스, 커뮤니티 등에서 매력적인 애니메이션 캐릭터 사진을 발견했으나, 작품 이름을 찾지 못해 몇 달 동안 갈증을 느끼는 유저들이 많습니다. (실제로 본인도 3달 전 발견한 최애 캐릭터의 출처를 찾지 못해 이 프로젝트를 기획함.)

### 1.3 이론적 배경 및 인용 논문

본 프로젝트는 다음 논문의 방법론을 기반으로 합니다:

> **AniWho: A Quick and Accurate Way to Classify Anime Character Faces in Images**  
> arXiv:2208.11012 (2022)

핵심 내용:
| 모델 | Top-1 정확도 | 비고 |
|------|------------|------|
| EfficientNet-B7 | **85.08%** | 최고 정확도 |
| EfficientNet-B0 | 83.46% | 균형형 |
| MobileNetV2 | 81.92% | 최고 속도 |

- Transfer learning (ImageNet pretrained) + fine-tuning on anime character faces
- Prototypical Networks로 few-shot 학습 가능성도 검증
- **본 프로젝트**: 분류(Classification) → 검색(Retrieval)으로 확장. ArcFace loss + FAISS 인덱스로 "본 적 없는 캐릭터"도 찾을 수 있도록 구성.

---

## 시스템 구조

```
쿼리 이미지
    ↓
EfficientNet-B7 Backbone  (AniWho 기반)
    ↓
Projection Head (512-dim)
    ↓
L2 Normalize
    ↓
FAISS IndexFlatIP  →  Top-K 유사 캐릭터 반환
                         ↓
                   캐릭터명 → 작품명 매핑
```

**학습**: ArcFace Loss (additive angular margin)로 같은 캐릭터 임베딩을 뭉치고 다른 캐릭터를 밀어냄.

---

## 프로젝트 구조

```
.
├── configs/
│   ├── model.yaml          # 백본, 임베딩 차원 설정
│   └── train.yaml          # 학습 하이퍼파라미터
├── data/
│   ├── raw/                # 수집한 원본 이미지
│   ├── processed/          # 전처리 완료 (캐릭터별 폴더 구조)
│   └── external/           # 외부 데이터셋 (Danbooru 등)
├── notebooks/
│   └── eda.ipynb           # 데이터 탐색
├── src/
│   ├── datasets/           # AnimeCharacterDataset
│   ├── models/             # EfficientNetEmbedder
│   ├── trainers/           # Trainer (AMP 지원)
│   ├── losses/             # ArcFaceLoss
│   ├── metrics/            # Top-K Accuracy, mAP
│   ├── utils/              # transforms, seed
│   └── inference/          # AnimeRetriever (FAISS)
├── scripts/
│   ├── train.py            # 학습 진입점 (Hydra)
│   ├── eval.py             # 검색 성능 평가
│   └── export.py           # FAISS 인덱스 저장
├── checkpoints/            # 학습된 모델 가중치
├── outputs/                # 로그, 시각화, 예측 결과
└── tests/
```

---

## 빠른 시작

### 환경 설정

```bash
pip install -e ".[dev]"
```

### 데이터 준비

`data/processed/` 아래 캐릭터별 폴더 구조로 정리:
```
data/processed/
  rem_re_zero/
    001.jpg
    002.jpg
  zero_two_darling/
    001.jpg
    ...
```

### 학습

```bash
python scripts/train.py
```

설정 오버라이드 예시 (Hydra):
```bash
python scripts/train.py training.epochs=50 model.backbone=efficientnet_b0
```

### FAISS 인덱스 빌드

```bash
python scripts/export.py --checkpoint checkpoints/epoch_030_loss0.1234.pt
```

### 평가

```bash
python scripts/eval.py --checkpoint checkpoints/epoch_030_loss0.1234.pt
```

### 추론 (Python API)

```python
from src.inference import AnimeRetriever
from src.models import EfficientNetEmbedder
import torch

model = EfficientNetEmbedder()
ckpt = torch.load("checkpoints/best.pt")
model.load_state_dict(ckpt["model_state"])

retriever = AnimeRetriever(model)
retriever.load("checkpoints/index.faiss")

results = retriever.search("query_character.jpg", top_k=5)
for character, score in results:
    print(f"{character}: {score:.4f}")
```

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| Top-1 Accuracy | 1순위 결과가 정답인 비율 |
| Top-5 Accuracy | 상위 5개 안에 정답이 있는 비율 |
| mAP | Mean Average Precision (검색 품질 종합) |

---

## 참고 문헌

- Zheng, H., et al. "AniWho: A Quick and Accurate Way to Classify Anime Character Faces in Images." arXiv:2208.11012 (2022).
- Deng, J., et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.
- Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML 2019.
