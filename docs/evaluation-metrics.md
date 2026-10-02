# 평가 지표 (룰베이스 vs 딥러닝 공통/개별)

이 프로젝트의 목표는 "모델 정확도 극대화"가 아니라 **4갈래 방법론(룰베이스/Mask R-CNN/ResNet/EfficientNet)을 공정하게 비교**하는 것이다. 그래서 지표도 "공통으로 똑같이 재는 것"과 "트랙 성격상 다르게 볼 수밖에 없는 것"을 나눠서 본다.

## 공통 지표 (4트랙 전부 동일한 방식으로 측정 — 공정 비교의 기준)

- **Accuracy** (Val/Test 각각) — 기본 지표지만 이것만으로는 부족함.
- **Confusion matrix + 클래스별 Precision/Recall/F1** — README에 이미 "헷갈릴 상대" 쌍(직선 원통형↔연속 테이퍼형, 연속 테이퍼형↔단차 테이퍼형)이 명시돼 있다. 단일 accuracy로는 어디서 틀리는지 안 보이므로, 사실상 가장 중요한 지표.
- **Macro-F1** — 클래스별로 동등하게 취급해서, 한 클래스만 유독 망가진 트랙을 accuracy 숫자 뒤에 숨기지 않게 함.
- **Val→Test 정확도 하락폭** — 이 프로젝트는 domain shift 평가(train=크롤링, test=직접 촬영)가 설계 목적이므로, 이건 부가 정보가 아니라 필수 지표다. 하락폭이 크다고 "나쁜 트랙"이 아니라 "그 트랙이 도메인 이동에 얼마나 취약한가"로 해석한다.

## 트랙별로 다르게 봐야 하는 지표

| 구분 | 지표 | 비고 |
|---|---|---|
| 공통(전 트랙) | 추론 속도 (ms/장) | 룰베이스 vs Mask R-CNN vs ResNet/EfficientNet 비교가 프로젝트 취지 중 하나 |
| DL 트랙만 (dl1/dl2/efficientnet) | epoch별 train/val loss·accuracy 커브 | 과적합 진단용. 룰베이스는 학습 루프 자체가 없어서 해당 없음 |
| 룰베이스만 | 임계값 민감도 | `MUG_HEIGHT_TO_DIAMETER_MAX`, `STEP_JUMP_RATIO_THRESHOLD` 같은 상수를 살짝 흔들었을 때 결과가 얼마나 튀는지. DL의 "학습 곡선"에 대응하는, 룰베이스만의 진단 지표 |

## 미해결: confidence 정의 불일치

`src/pipeline.py`는 4트랙 공통으로 `classify(image) -> {shape, confidence, explanation}` 형태를 반환하도록 설계돼 있는데, 트랙마다 confidence의 의미가 다르다.

- **DL 트랙(dl1/dl2/efficientnet)**: softmax 확률이 자연스러운 confidence.
- **룰베이스**: `classify_shape()`가 지금 반환하는 건 `height_diameter_ratio`, `bottom_top_ratio` 같은 원시 비율값뿐, 진짜 confidence가 없음.

**해결 방향(미구현)**: 룰베이스는 "판정 임계값까지 얼마나 여유 있게 넘겼는지"를 0~1로 정규화해서 confidence처럼 쓸 수 있다. 예를 들어 mug 판정이면 `1.5`(임계값)와의 거리를, straight 판정이면 `0.95`와의 거리를 정규화하는 식. 이래야 pipeline.py 단계에서 트랙 간 confidence를 같은 스케일로 비교/표시할 수 있다.

## 적용 방법

- 새 실험 결과를 `docs/experiment-log.md`에 기록할 때, 단순 accuracy뿐 아니라 위 공통 지표(특히 confusion matrix, Val→Test 하락폭)를 같이 남긴다.
- `notebooks/03_model_comparison.ipynb`에서 4트랙을 비교할 때 이 문서의 공통 지표 기준으로 표/그래프를 맞춘다.
- 룰베이스 confidence 정규화 함수는 아직 미구현 — 필요해지면 `src/rule_based/shape_classifier.py`에 추가.
