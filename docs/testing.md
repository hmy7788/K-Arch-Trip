# 테스팅 가이드

이 프로젝트에는 자동화된 유닛 테스트(`tests/`)와, 각 트랙·데모 앱이 실제로 돌아가는지 확인하는 수동 재현 절차가 섞여 있다. 전체 기능을 점검할 때 아래 순서를 따른다.

## 1. 유닛 테스트 (pytest)

```bash
python -m pytest tests/ -q
```

룰베이스 `shape_classifier.py`의 합성 마스크 10종(머그형/직선/연속테이퍼/단차테이퍼 + 손잡이 변형, 폭 프로파일, 손잡이 검출)을 검증한다. 다른 트랙을 건드린 작업이어도 커밋 전에는 항상 이걸 먼저 돌린다 — 회귀 확인용.

## 2. 딥러닝 트랙 — `--eval-only` 재평가

재학습 없이 기존 체크포인트로 Test1(Val)/Test2(실촬영) 정확도·Macro-F1을 다시 계산한다. 아래 수치는 공식 체크포인트 기준 참고값(`docs/experiment-log.md`와 동일) — 실행 결과가 이 범위에서 벗어나면 체크포인트나 데이터 분할이 깨진 것이다.

```bash
# ResNet-18 (TTA 포함, data/test1 158장 기준)
python src/deep_learning/dl2_resnet/train.py --model resnet18 --no-freeze-backbone --eval-only --tta
# 기대값: Val 96.9% / Test 82.3%, Macro-F1 0.818

# ResNet-50
python src/deep_learning/dl2_resnet/train.py --model resnet50 --no-freeze-backbone --eval-only

# EfficientNet-B0 (data/test1_orientation_backup 126장 기준)
python src/deep_learning/efficientnet/train.py --eval-only
# 기대값: Val 96.9% / Test 88.1%, Macro-F1 0.879

# ConvNeXt-Tiny
python src/deep_learning/convnext/train.py --eval-only
# 기대값: Val 98.4% / Test 86.5%, Macro-F1 0.867

# ViT 완전 베이스라인
python src/deep_learning/vit/train.py --baseline --eval-only --test-root data/test1_orientation_backup
# 기대값: Val 98.4% / Test 85.7%, Macro-F1 0.856

# ViT 실험 B (taehyun 설계)
python src/deep_learning/vit/train.py --experiment B --eval-only --test-root data/test1_orientation_backup
# 기대값: Val 99.2% / Test 73.8%, Macro-F1 0.729

# Custom CNN (교육용 대조군, 4트랙 공식 비교 제외)
python src/deep_learning/custom_cnn/train.py --eval-only
```

**주의**: `--eval-only`는 eval 모드(dropout 없음, 고정 가중치)라서 재실행해도 결과가 100% 동일해야 한다. 숫자가 바뀐다면 체크포인트가 덮어써졌거나 데이터 분할 알고리즘이 트랙 간에 어긋난 것이니 바로 확인할 것 — 특히 `CLASSES` 순서(`["straight", "taper_smooth", "taper_step", "mug"]`)가 모든 트랙에서 동일한지부터 본다.

## 3. 룰베이스 / Mask R-CNN

```bash
# 룰베이스 — 테스트1(Val, 같은 도메인 분할)
python src/rule_based/evaluate_test.py --source test1
# 기대값: 63.0%(80/127), Macro-F1 0.610

# 룰베이스 — 테스트2(data/test1, 158장, 직접 촬영)
python src/rule_based/evaluate_test.py --source test2
# 기대값: 18.4%(29/158), Macro-F1 0.139 — GrabCut 때문에 느림(158장에 수 분 소요), 느긋하게 기다릴 것

# Mask R-CNN 제로샷 vs 파인튜닝 비교
python src/deep_learning/dl1_maskrcnn/evaluate_test.py --test-root data/test
```

## 4. Grad-CAM (판정 근거 시각화)

```bash
python src/deep_learning/dl2_resnet/gradcam.py --split test2 --per-class 4 --model resnet18
python src/deep_learning/vit/gradcam.py --split test2 --per-class 4
```

`--split test1`로 바꾸면 Val(같은 도메인) 기준 히트맵이 나온다. `--per-class`는 2 이상으로 쓸 것 — matplotlib `subplots`가 `squeeze=False`로 고정돼 있어 1도 동작은 하지만(과거엔 버그로 죽었음, 2026-10 수정됨), 그리드가 한 줄뿐이라 보기 불편하다.

이 스크립트들은 `reports/figures/<track>/gradcam_test1_val.png` / `gradcam_test2.png`를 덮어쓴다. 공식 피규어를 재현할 때는 기본값(`--per-class 4`, `--seed 0`)을 그대로 써야 커밋된 이미지와 동일한 결과가 나온다 — 다른 값으로 실행하면 `git status`에 해당 PNG가 "수정됨"으로 뜨니, 확인 목적이었다면 `git checkout -- <경로>`로 되돌릴 것.

## 5. 데모 앱 헬스체크

두 데모 앱 모두 모델을 실제로 로드하는지까지 확인하려면 서버를 띄우고 헬스체크 엔드포인트를 쳐본다(브라우저 UI 동작까지 보려면 직접 열어봐야 함 — 아래는 "안 죽고 뜨는지"만 확인하는 스모크 테스트).

```bash
# Streamlit (K-Arch Trip)
python -m streamlit run app/demo.py --server.headless true --server.port 8610 &
curl http://localhost:8610/_stcore/health   # "ok" 가 나와야 함

# K-Trip FastAPI
cd k_trip_ios && python -m uvicorn server:app --port 8611 &
curl http://127.0.0.1:8611/api/health       # {"ready": true, "model": {...}} 가 나와야 함
```

둘 다 `checkpoints/model/meta.json` (Streamlit) / `k_trip_ios/checkpoints/meta.json` (K-Trip)으로 모델을 교체할 수 있다. 헬스체크 응답의 `model.arch`로 어떤 체크포인트가 물려 있는지 확인한다.

**주의**: 백그라운드로 띄운 서버는 터미널 세션이 바뀌면 `kill %1`이 안 먹을 수 있다 — 테스트 끝나면 `netstat -ano | grep LISTENING`으로 포트를 쥔 PID를 찾아 직접 종료할 것(그대로 두면 GPU 메모리를 계속 물고 있음).

## 알려진 함정

- **데이터 분할 재현성**: 모든 트랙이 `data/preprocess`를 `CLASSES` 알파벳이 아닌 고정 순서(`straight, taper_smooth, taper_step, mug`)·`seed=42`로 분할해야 Test1(Val) 127장이 트랙 간에 동일하게 재현된다. 새 트랙을 추가할 때 가장 흔한 실수.
- **테스트셋 버전 차이**: ResNet은 기본 `data/test1`(158장), 나머지 신규 트랙(EfficientNet/ConvNeXt/ViT)은 `data/test1_orientation_backup`(126장, 32장 정제됨)을 기본값으로 쓴다. 트랙 간 Test2 숫자를 직접 비교할 때는 같은 `--test-root`로 맞춰서 재평가해야 공정하다.
- **룰베이스 평가는 느리다**: GrabCut이 이미지당 1~2초 걸려서 158장 기준 수 분 소요. 백그라운드 실행 권장.
