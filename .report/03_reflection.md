# 3. 시행착오 및 성찰

## 3.1 문제 발생 및 해결 과정

### 문제 1: EfficientNet-B4가 B0보다 낮은 정확도

더 깊고 넓은 B4(17.6M params)가 더 작은 B0(4.0M params)보다 낮은 성능(69.69% vs 74.56%)을 기록했다.

**원인 분석**: 클래스당 평균 ~95장의 소규모 데이터(총 1,432장)에서 B4의 파라미터 수가 과도해 과적합이 발생했다. Val Loss가 학습 중반부터 B0 대비 높게 유지된 것이 이를 뒷받침한다.

**해결 방향**: 데이터 증강 강화(RandomErasing, Mixup 등) 또는 더 강한 정규화(Dropout, weight decay 증가) 적용이 필요하다. 최종 프로덕션 모델로 B0를 선택하거나 더 많은 데이터를 수집해야 한다.

### 문제 2: MPS 환경에서 학습 속도 편차

ViT-B/16은 260초인 반면 EfficientNet-B4는 4,055초로 15배 이상 차이가 났다. EfficientNet의 depthwise separable convolution이 MPS에서 최적화가 덜 된 것으로 추정된다. 반면 ViT는 행렬 곱 위주 연산이라 MPS에서 효율적이었다.

### 문제 3: Phase 1에서 ViT 수렴이 매우 빠름

ViT-B/16은 Phase 1의 첫 에폭부터 val_acc=0.7143으로 빠르게 수렴했다. EfficientNet이 Phase 1에서 0.29~0.47에 머문 것과 대조적이다. ViT가 ImageNet 사전학습에서 더 범용적인 특징을 학습했기 때문으로 보인다.

## 3.2 학습 성찰

**모델 크기 ≠ 성능**: 파라미터가 많다고 항상 좋은 결과가 나오지 않는다. 데이터 규모에 맞는 모델 선택이 중요하다. 소규모 데이터에서는 작고 효율적인 모델(B0)이 대형 모델(B4)을 이길 수 있다.

**Transformer의 강점 재확인**: ViT-B/16이 가장 높은 정확도와 가장 빠른 학습 속도를 동시에 달성했다. 전이학습 품질과 MPS 하드웨어 활용 면에서 ViT가 우수했다. 다만 85.8M params는 엣지 배포에 불리하므로, 실서비스에는 EfficientNet-B0(4.0M)가 더 현실적인 선택이다.

**Retrieval vs Classification**: 이번 실험은 분류 모델로 진행했으나 최종 시스템은 ArcFace + FAISS 기반 retrieval 구조다. 분류 정확도가 곧 retrieval 성능은 아니며, 실제 검색 품질은 임베딩 공간의 밀집도(clustering)에 달려 있다. 향후 ArcFace loss로 학습한 임베딩의 검색 정확도(Recall@K) 측정이 필요하다.
