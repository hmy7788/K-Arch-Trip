# 실험 결과 로그

4갈래 트랙(룰베이스 / Mask R-CNN / ResNet / EfficientNet)의 실험 결과를 트랙별로 기록한다. **검증 대상은 시스템**이므로, 정확도 숫자만이 아니라 구현 난이도·추론 속도·실패 패턴도 함께 남긴다.

## 현재 순위 요약 (2026-09-16, `data/test1` 158장 기준)

사용자가 직접 검수한 실제 촬영 Test 세트(클래스당 27~50장, 클래스: straight/taper_smooth/taper_step/mug)로 전부 재평가한 최신 스냅샷(이전 `data/test` 104장 기준 스냅샷을 대체). 아래 표만 보면 지금 가장 나은 파이프라인이 뭔지 바로 알 수 있다 — 아래쪽 트랙별 표는 그 결과에 이르기까지의 실험 과정(시행착오 포함) 기록.

| 순위 | 트랙 | Test 정확도 | Macro-F1 | 비고 |
|---|---|---|---|---|
| 🥇 1 | **ResNet-18 + TTA** (`--no-freeze-backbone --tta`) | **81.6%** (129/158) | **0.813** | 스케일 3종x좌우반전 6-view 평균, 재학습 없이 기존 체크포인트로 추론만 바꾼 것. 체크포인트: `checkpoints/resnet18_shape.pth` |
| 2 | ResNet-50 (`--no-freeze-backbone`) | 81.0% (128/158) | 0.809 | TTA 미적용 기준. Val→Test 하락폭(18.2%p)이 18보다 커서 더 과적합하는 경향 — 굳이 50을 쓸 이유는 약함(아래 표 참고) |
| 3 | ResNet-18 (`--no-freeze-backbone`, TTA 없이) | 81.0% (128/158) | 0.803 | 백본까지 fine-tuning, 차등 LR. Val→Test 하락폭 15.8%p로 50보다 안정적. **TTA 켜면 1위로 올라감(위 항목)** |
| 4 | Mask R-CNN (제로샷) | 43.7% (69/158) | 0.466 | COCO 사전학습 그대로, fine-tuning 안 함(`src/deep_learning/dl1_maskrcnn/segment.py`) |
| 5 | Mask R-CNN (파인튜닝) | 36.7% (58/158) | 0.351 | 룰베이스 pseudo-label로 fine-tuning — **실사용 안 함**, 실패 사례로 기록만 유지 |
| 6 | 룰베이스 | 18.4% (29/158) | 0.139 | `src/rule_based/`. 폭 계산을 손잡이-몸통 분리 방식으로 개선(9/16)했지만 여전히 대부분 mug로 쏠림 — segmentation 단계까지 손댄 추가 시도는 오히려 하락해서 롤백(9/16, 아래 표 참고) |
| - | EfficientNet-B0 | 미착수 | - | |

참고: ResNet-50에는 아직 TTA를 안 붙여봤음 — 50에도 붙이면 순위가 다시 바뀔 수 있음(미착수).

정확도와 macro-F1 순위가 대체로 일치한다는 점이 중요하다 — 즉 룰베이스·Mask R-CNN 파인튜닝의 낮은 정확도가 "머그형에만 몰아서 찍어 숫자만 맞춘" 클래스 불균형 편법이 아니라, **전 클래스에 걸쳐 고르게 성능이 나쁘다**는 뜻(Macro-F1은 표본 수와 무관하게 클래스별 F1을 동일 가중치로 평균내므로). 룰베이스는 여전히 mug precision이 극히 낮은 채(대부분 오답이 mug로 쏠림) recall만 높은 구조적 편향을 보인다. ResNet-18 vs 50은 **정확도로는 우열이 안 갈리고 Val→Test 하락폭(과적합 정도)으로만 갈린다** — 파라미터를 늘린다고 이 데이터 규모(train 515장)에서 더 나아지지 않는다는 것도 이번 비교의 중요한 결론.

**핵심 교훈**: 이 프로젝트의 제일 큰 domain shift 원인은 **촬영 각도(원근 왜곡)** — 위에서 내려다보고 찍으면 직선이 테이퍼져 보이고 키가 눌려 보여서, 폭 프로파일 같은 **기하학적 규칙에 의존하는 트랙(룰베이스, 그리고 그 규칙을 공유하는 Mask R-CNN)일수록 크게 무너진다.** ResNet처럼 **학습된 시각 패턴**(색상·질감·손잡이 모양·맥락 등)을 쓰는 방식이 이 왜곡에 훨씬 강하다는 게 이번 실험들로 반복 확인됨. Grad-CAM으로 봐도 ResNet은 배경이 아니라 물체 본체·손잡이에 정확히 집중하고 있었다(`reports/figures/resnet18/gradcam.png`).

---

새 실험을 기록할 때는 각 트랙 표에 행을 추가한다. 형식:

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| YYYY-MM-DD | 이름 | 예: ResNet-18 baseline, backbone freeze | 0.00 | 0.00 | 혼동 클래스, 특이사항 등 |

---

## 룰베이스 (`src/rule_based/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-09-14 | Claude | 최초 구현: get_mask/compute_width_profile/detect_handle/classify_shape + 합성 마스크 10종 pytest | N/A (실데이터 없음) | N/A | 합성 마스크 4종(+손잡이 변형) 전부 올바른 라벨로 분류(10/10 테스트 통과). 실사진 확보 전까지는 정확도 숫자 의미 없음 — get_mask의 세그멘테이션 견고성이 다음 병목 |
| 2026-09-14 | Claude | 실사진 4장(클래스당 1장) 수동 테스트 — 연속테이퍼형/직선원통형/단차테이퍼형/머그형 각 1장 | 2/4 정확(50%, n=4로 통계적 의미 없음) | - | **연속테이퍼형·직선원통형**: 단색 배경+고대비 → 마스크 깨끗, 정확히 분류. **단차테이퍼형**: 그림자가 마스크에 붙고 아래쪽 단차 부분이 통째로 누락됐는데도 라벨은 우연히 맞음(height/diameter=1.51, bottom/top=0.92 — 둘 다 임계값 바로 옆이라 불안정). **머그형**: 나무 배경과 코르크 받침 색이 비슷해 마스크가 중간에서 잘리고 손잡이도 누락 → straight로 오분류. get_mask()의 배경분리 한계가 예상대로 실제 데이터에서 터짐 — GrabCut 등 개선 필요 (다음 실험에서 처리) |
| 2026-09-14 | Claude | get_mask()에 GrabCut 정제 추가(Otsu 결과를 시드로 사용) 후 같은 4장 재테스트 | 2/4 정확(동일) | - | 단차테이퍼형의 위험한 경계값이 완화됨(h/d 1.51→1.58, b/t 0.92→0.89, 임계값에서 더 멀어짐) + 연속테이퍼형 마스크 디테일 향상. **머그형(코르크=나무 바닥 색 충돌)과 단차테이퍼형(그림자 융착)은 GrabCut으로도 해결 안 됨** — 색상 기반 알고리즘의 근본 한계, 촬영 조건(배경 대비·그림자 없는 조명) 문제에 가까움. 이런 어려운 케이스는 결국 Mask R-CNN(dl1) 트랙이 처리하도록 설계된 부분 |
| 2026-09-15 | Claude | `data/raw2`(수집·미검증 상태)에서 클래스당 무작위 10장(n=40, seed=42) 샘플링 후 `classify_shape()` 일괄 실행 — `notebooks/rule_based_raw2_sample.ipynb` | 폴더 라벨 기준 일치율: straight 100%(10/10), mug 100%(10/10), taper_smooth 60%(6/10), **taper_step 30%(3/10)**, 전체 72.5%(29/40) | - | **taper_step 6/10이 mug로 오분류** — 원인 확정: `get_mask()`가 몸통을 놓치고 금속 뚜껑/테두리만 마스크로 잡는 경우가 반복됨 → 높이가 과소측정되어 height/diameter≤1.5(mug 조건)를 충족해버림. 그림자/색충돌 케이스(9/14 실험)와 별개로, **taper_step 특유의 "몸통 놓침" 실패 모드**가 구조적으로 반복됨을 n=40에서 확인. 마스크가 몸통까지 제대로 잡힌 3장은 전부 정확히 분류됨 — 로직 자체는 문제없고 세그멘테이션이 병목. taper_smooth의 오분류(mug 2, taper_step 2)는 원인 미분석. **주의**: raw2 라벨은 미검수 상태라 일치율에 "엉뚱한 이미지가 섞여서 생긴 불일치"도 일부 포함될 수 있음(둘을 분리 못 함) |
| 2026-09-15 | Claude | `get_mask()`에 Canny 엣지 기반 마스크(`_rough_mask_canny`)를 Otsu와 OR로 합쳐 GrabCut 시드로 사용하도록 개선 후 동일 40장 재실행 | straight 90%(9/10), taper_smooth 50%(5/10), **taper_step 60%(6/10)**, mug 100%(10/10), 전체 **75.0%(30/40)** | - | 원인 진단: 실패 6장 중 3장(101/65/83.jpg)이 흰색 텀블러+흰색 배경이라 Otsu 전경 비율이 1.3~5.6%까지 떨어짐(명도 대비 자체가 없음) — Canny는 명도 절대값이 아니라 옅은 경계선을 보므로 이 케이스에 강함. **taper_step 30%→60%로 2배 개선, 전체도 순개선(72.5%→75%)**. 단, straight·taper_smooth에서 각 1건씩 새 오분류 발생(Canny가 배경 텍스처를 같이 잡는 부작용으로 추정, 미분석) — trade-off 있는 개선. 여전히 실패: 70.jpg(클로즈업 사진, 데이터 문제이지 알고리즘 문제 아님), 63.jpg(복잡한 야외 배경) |
| 2026-09-15 | Claude | `classify_shape()` 임계값을 실측(data/raw2 445장, Mask R-CNN 마스크 기준)으로 재조정 — `notebooks/threshold_calibration.ipynb`. 각 갈림길을 이진분류로 놓고 정확도 최대화 지점 탐색: MUG_HEIGHT_TO_DIAMETER_MAX 1.5→1.5(유지), STRAIGHT_BOTTOM_TOP_RATIO_MIN 0.95→**0.92**, STEP_JUMP_RATIO_THRESHOLD 0.4→**0.29** | Mask R-CNN 마스크 기준(445장): 84.5%→**88.0%**. 룰베이스 마스크 기준(445장): 66.3%→65.8%(거의 변화 없음) | - | Mask R-CNN 파이프라인에서 taper_step 75.7%→88.8%로 크게 개선(전체 3.5%p↑), taper_smooth는 90.0%→83.0%로 소폭 하락(trade-off). 룰베이스는 Mask R-CNN 마스크로 캘리브레이션한 값이라 전이가 완벽하진 않지만(-0.5%p) 손해가 거의 없어 공통 채택. **부수 효과**: 합성 테스트 `test_taper_smooth_with_handle_stays_taper_smooth`가 새 임계값에 걸려 실패 → 원인은 임계값이 아니라 테스트용 손잡이 돌기가 폭 구간 경계에 걸쳐 median smoothing으로 안 지워지는 인공적 단차를 만든 것으로 확인, `synthetic.py`에서 손잡이 위치를 구간 안쪽으로 조정해 해결 (`docs/troubleshooting.md` 참고). `classify_shape()`가 `step_ratio`를 항상 반환하도록 리팩터링(이전엔 taper 분기에서만 계산됨) |
| 2026-09-15 | Claude | `data/test` 확대판(104장, test2 병합)으로 재평가 — `src/rule_based/evaluate_test.py` 신규 작성(다른 트랙과 같은 패턴: confusion matrix를 `reports/figures/rule_based/`에 저장) | - | **11.5%(12/104)** | 이전 56장 기준(18%)보다도 낮아짐 — 표본이 늘면서 실제 약점이 더 뚜렷하게 드러남. 클래스별: straight 1/38(2.6%), taper_smooth 1/22(4.5%), taper_step 4/34(11.8%), mug 6/10(60.0%). **straight/taper_smooth/taper_step 거의 전부 mug로 쏠림**(26/38, 18/22, 28/34) — 원인은 기존과 동일(촬영 각도 원근 왜곡으로 키가 눌려 보여 mug 경계(h/d≤1.5)를 넘어버림, docs/troubleshooting.md 9/15 항목). 4트랙 중 유일하게 Test에서 mug가 제일 정확한 클래스(다른 트랙은 mug가 오히려 약하거나 평범함) — 룰베이스의 판정 로직 자체가 "애매하면 mug로 판정"하는 구조적 편향을 갖고 있음을 시사 |
| 2026-09-16 | Claude | mug 붕괴 원인을 실제 이미지로 직접 진단(taper_step 손잡이 사진, straight 하향각 사진에 `get_mask`/`classify_shape` 수동 실행) 후 `compute_width_profile()`을 "구간 내 min-max 폭" → "행별 최장 연속 전경 구간 폭의 중앙값"으로 수정(`shape_classifier.py`). 사용자가 직접 검수한 `data/test1`(158장, straight/taper_smooth/taper_step/mug 각 50/41/40/27장)으로 재평가 | - | **18.4%(29/158)**, Macro-F1 0.139 | 합성 마스크 pytest 10개 전부 통과(손잡이가 몸통에 맞닿은 합성 케이스는 연속 구간이 하나로 합쳐져 이전과 동일하게 동작 — 회귀 없음). 클래스별: straight 5/50(10.0%), taper_smooth 0/41(0.0%), taper_step 4/40(10.0%), mug 20/27(74.1%) — **여전히 나머지 3클래스 대부분 mug로 쏠림**(37/50, 34/41, 34/40). 진단 결과 원인은 폭 계산 로직이 아니라 그 앞 단계: taper_step 손잡이 사진에서 `get_mask()`의 최종 정제 단계(`_rough_mask_canny`의 15x15 morphological close)가 손잡이의 그립 구멍을 GrabCut 이전에 이미 메워버려서, 구멍이 배경으로 살아남는 경우가 거의 없음(실측: FILLED 전후 전경 픽셀 수 완전히 동일, 즉 애초에 뚫린 적이 없었음) — 그래서 폭 계산을 고쳐도 효과가 제한적. `data/test`(11.5%) 대비 정확도·Macro-F1 둘 다 개선됐지만(테스트셋이 달라 완전한 비교는 아님), 근본 원인(segmentation 단계의 구멍 메움)은 아직 해결 안 됨 — closing 커널을 줄이면 저대비 배경 분리가 다시 약해지는 trade-off가 있어 추가 검토 필요 |
| 2026-09-16 | Claude | (**시도했다가 되돌림**) segmentation 단계에서 손잡이 구멍을 실제로 살리는 시도 3단계: ① `RETR_CCOMP`로 구멍(자식 컨투어)을 감지해 배경으로 유지하는 `_fill_largest_contour_preserving_holes()` 추가 → Canny 경로(엣지-선 입력)에 적용했더니 마스크가 반전되는 회귀 발견, Canny는 원래 방식으로 되돌림. ② Otsu 경로에만 적용 + union(OR) 특성상 Canny가 구멍을 다시 메우는 문제 발견 → Otsu가 판단한 구멍을 union에서 강제로 다시 파내는 로직 추가. ③ 그래도 GrabCut의 매끄러움(smoothness) 항이 고립된 배경 섬을 주변 전경 쪽으로 끌어당겨 구멍을 다시 삼키는 것 확인 → `GC_BGD`(확정 배경, 절대 안 뒤집힘) 하드 제약으로 전달하도록 `_grabcut_refine()` 확장 | - | **data/test1: 17.1%(27/158), Macro-F1 0.127 — 이전(18.4%/0.139)보다 오히려 하락** | 진단했던 특정 사진(손잡이 상하 연결부가 몸통에 실제로 맞닿아 있어 그 부분은 애초에 살릴 수 없는 진짜 구조)은 부분적으로만 개선되고 최종 분류는 그대로 mug였음. 158장 전체로는 순 하락 — 아마 다른 사진들(반사광·하이라이트 등)에서 진짜 물체 표면을 "구멍"으로 오인해 배경으로 파내는 부작용이 이득보다 컸던 것으로 추정(정밀 원인 미분석). **`shape_classifier.py`는 이 변경 전(median 폭 계산 유지) 상태로 롤백함.** 세그멘테이션 단계를 더 손대는 접근은 위험 대비 이득이 낮아 보임 — 다음에 시도한다면 "어떤 사진에서 새로 나빠졌는지" 먼저 특정하고 시작할 것 |

| 2026-09-16 | Claude | **"테스트1/테스트2" 프레이밍을 룰베이스에도 적용** — `evaluate_test.py`에 `--source test1` 추가. dl2_resnet/train.py의 stratified_split()과 동일한 알고리즘(같은 CLASSES 순서·정렬·seed=42)을 복제해서 `data/preprocess`에서 ResNet의 Val과 **완전히 동일한 127장**을 재현, 룰베이스로 평가. 테스트2는 `data/test1`(158장)로 재확인 | - | **테스트1: 63.0%(80/127), Macro-F1 0.610** / **테스트2: 18.4%(29/158), Macro-F1 0.135**(기존과 오차범위 내 동일) | ResNet-18(TTA, 96.9%→82.3%, 하락폭 14.6%p)과 나란히 비교: 룰베이스는 63.0%→18.4%로 **하락폭이 44.6%p**나 돼서 절대 수치도 낮고 도메인 시프트에 더 취약하다는 게 숫자로 재확인됨. 테스트1(크롤링 이미지, 대부분 정면 카탈로그 컷)에서는 그럭저럭 쓸만하다는 것도 확인 — 즉 룰베이스가 "아예 안 되는" 게 아니라 "정면 사진 전제가 깨지면 무너지는" 방식이라는 원래 가설이 정량적으로 뒷받침됨 |

## Mask R-CNN (`src/deep_learning/dl1_maskrcnn/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-09-15 | Claude | `segment.py` 최초 구현(COCO 사전학습 `maskrcnn_resnet50_fpn_v2`, 제로샷·fine-tuning 없음) + `rule_based_raw2_sample.ipynb`와 동일한 40장(seed=42)에 룰베이스와 나란히 비교 — `notebooks/maskrcnn_vs_rule_based.ipynb` | - (raw2, 미검증 라벨 기준 참고용) | - | **전체 75%(룰베이스) → 85%(Mask R-CNN)**. 클래스별: straight 90→100%, taper_smooth 50→70%, **taper_step 60→80%**(목표했던 개선), mug 100→90%(소폭 회귀). 추론 속도는 GPU(RTX 4050)에서 평균 150ms/장으로 룰베이스(CPU, 503ms/장, GrabCut이 병목)보다도 빠름. cup/bottle/vase/wine glass/bowl 중 점수 높은 COCO 검출을 사용 — 텀블러가 COCO 클래스에 없어서 근접 카테고리로 대체 |
| 2026-09-15 | Claude | 룰베이스 마스크를 pseudo-label 삼아 fine-tuning 시도 (배경/텀블러 2클래스, 평가용 40장은 학습에서 제외, `classify_shape()`가 폴더 라벨과 맞은 것만 채택) — `notebooks/maskrcnn_finetune_rulebase_pseudolabels.ipynb`. 학습 266장(3 epoch, 1197s, loss 0.376→0.140), GPU RTX 4050 | 동일 40장(held-out) 재평가: 78%(31/40) | - | **제로샷(85%)보다 오히려 나빠짐(78%) — 우려했던 리스크가 실제로 발생.** straight 100→80%, taper_smooth 70→60%, taper_step 80→70% 전부 회귀, mug만 90→100%(가장 많이 채택된 클래스라 치우침 발생). 원인: (1) 클래스별 pseudo-label 채택 수가 심하게 불균형(mug 88 / straight 81 / taper_smooth 64 / **taper_step 33** — 룰베이스가 약한 클래스일수록 채택 수도 적어져서 학습 데이터가 mug 쪽으로 편향), (2) 룰베이스 마스크의 거친 경계가 COCO 학습으로 얻은 정밀한 마스크 품질을 깎아먹은 것으로 추정. **결론: 이 체크포인트는 폐기, dl1 트랙은 제로샷 버전(`segment.py`의 기본 `load_model()`)을 계속 사용.** 체크포인트는 `checkpoints/maskrcnn_finetuned_rulebase_pseudolabels.pth`에 남겨두되 실사용 안 함(참고 기록용) |
| 2026-09-15 | Claude | 위 파인튜닝 체크포인트를 폐기하지 않고 `data/test`(실제 촬영 104장)에서 제로샷과 직접 비교 — `src/deep_learning/dl1_maskrcnn/evaluate_test.py` | - | **제로샷 46.2%(48/104) vs 파인튜닝 15.4%(16/104)** | **raw2(85%→78%)보다 실제 Test에서 훨씬 더 크게 무너짐(3배 차이).** Confusion matrix 확인: 파인튜닝 모델이 거의 전부 "mug"로만 예측(straight 24/38, taper_smooth 19/22, taper_step 26/34가 mug로 쏠림) — 학습 데이터의 mug 편향(위 항목 참고)이 domain shift 상황에서 더 극단적으로 드러남(class collapse). **파인튜닝을 다시 시도해볼 이유가 없다는 결론을 재확인** — 진짜 마스크 정답 없이는 근본적으로 개선 안 됨. 그래프: `reports/figures/maskrcnn/{zeroshot,finetuned}_test_confusion_matrix.png` |

| 2026-09-16 | Claude | 사용자가 직접 검수한 `data/test1`(158장)로 제로샷 vs 파인튜닝 재비교 — 룰베이스와 공유하는 `compute_width_profile()` 수정(9/16, 손잡이-몸통 분리 방식) 효과가 이 트랙에도 반영되는지 확인 겸 | - | **제로샷 43.7%(69/158, Macro-F1 0.466) vs 파인튜닝 36.7%(58/158, Macro-F1 0.351)** | 파인튜닝이 `data/test`(104장) 기준 15.4%에서 36.7%로 크게 올라감 — 공유 폭 계산 수정 효과가 일부 반영된 것으로 보임(class collapse가 다소 완화). 제로샷은 46.2%→43.7%로 소폭 하락(테스트셋 차이, 큰 의미 없는 변동). **여전히 파인튜닝 < 제로샷이라 "파인튜닝 실사용 안 함" 결론은 그대로 유지** |

## ResNet-18/50 (`src/deep_learning/dl2_resnet/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-09-15 | Claude | `train.py` 최초 구현 — ResNet-18 ImageNet 사전학습, 백본 freeze(FC layer만 학습), `data/preprocess`(642장, 강한 신호 제외) 층화 80/20 분할, 증강에 `RandomPerspective` 포함(촬영 각도 왜곡을 직접 겨냥), 20 epoch, GPU RTX 4050(145s) | 91.3%(best epoch 12) | 58.9%(33/56, data/test 초기 56장) | **압도적 개선**: 같은 Test 세트에서 룰베이스 18%, Mask R-CNN 39% 대비 두 배 가까이 높음 — 기하학적 규칙(`classify_shape`) 대신 학습된 시각 패턴을 쓰는 게 촬영 각도 왜곡에 훨씬 강하다는 가설이 실측으로 확인됨. Test에서 mug만 1/6(17%)로 급락 — 표본이 6장뿐이라 노이즈 의심. **주의: 이 체크포인트는 이후 사고로 유실됨(아래 항목 참고), 수치는 참고용** |
| 2026-09-15 | Claude | `data/test`에 `data/test2` 48장 병합(104장, mug 10장으로 확대) 후 재평가하려다, **재학습 프로세스를 도중에 강제종료했는데 1 epoch째 체크포인트 저장이 먼저 끝나버려 기존 91.3% 체크포인트가 그걸로 덮어써짐** — `--eval-only`로 불러온 val_acc가 80.3%로 나와서 발견. `train.py`에 안전장치 추가(학습 중에는 `.tmp` 파일에만 저장, 전체 epoch가 끝까지 성공했을 때만 `os.replace`로 최종 반영 — 중간에 죽어도 기존 체크포인트 안전) 후 처음부터 재학습 | **89.8%**(best epoch 18) | **64.4%**(67/104, data/test 104장 — mug 포함 확대판) | mug 정상화 확인: Test에서 5/10(50%)로 회복 — 이전 1/6(17%)이 정말 표본 노이즈였음이 확인됨. 클래스별 Test 정확도: straight 21/38(55%), taper_smooth 15/22(68%), **taper_step 26/34(76%, 4트랙 중 처음으로 taper_step이 제일 강한 클래스)**, mug 5/10(50%). Val→Test 하락폭 25.3%p로 이전 56장 기준(32.4%p)보다도 줄어듦 — 표본이 커지니 추정치가 더 안정적. 그래프/체크포인트는 이 실행 결과로 덮어써짐(위 항목의 수치는 재현 불가) |
| 2026-09-15 | Claude | `--no-freeze-backbone`(백본까지 fine-tuning) 실험. 백본은 낮은 LR(`args.lr*0.1`), 새 FC layer는 원래 LR을 쓰는 차등 학습률 적용(우선 `checkpoints_experiment/`에 저장해 기존 체크포인트 보호 후 비교, 결과가 확실히 나아서 공식 체크포인트로 승격) | **96.9%**(best epoch 12/14) | **79.8%**(83/104) | **4트랙·모든 실험 통틀어 최고 기록.** 프리즈 버전(Val 89.8%→Test 64.4%) 대비 Val·Test 둘 다 큰 폭 상승, Val→Test 하락폭도 25.3%p→**17.0%p**로 축소 — domain shift 자체에 더 강해짐. 학습 시간은 145s로 프리즈 버전과 거의 동일(이 작은 데이터셋에서는 역전파 비용이 병목이 아니었음). Test confusion matrix: straight 29/38(76%), taper_smooth 13/22(59%), taper_step 32/34(94%), mug 9/10(90%) — 전 클래스 고르게 개선. train_acc가 97~99%로 val보다 훨씬 높아 과적합 신호는 있으나(train_loss↓ 지속, val_loss 진동), best-epoch 체크포인트 저장 덕에 실사용엔 문제 없음. **이후 dl2_resnet 실험은 기본적으로 이 옵션을 쓸 것.** |
| 2026-09-16 | Claude | 사용자가 직접 검수한 `data/test1`(158장, straight/taper_smooth/taper_step/mug 각 50/41/40/27장)로 기존 체크포인트(`--eval-only`) 재평가 | 96.9%(동일 체크포인트) | **81.0%(128/158)**, Macro-F1 0.803 | `data/test`(104장) 기준(79.8%, Macro-F1 0.769)보다 소폭 개선. 클래스별: straight 86.0%, taper_smooth 61.0%(4트랙 중 가장 약한 클래스로 재확인), taper_step 97.5%, mug 77.8%. 재학습 없이 큐레이션된 평가셋으로만 재확인한 결과라 체크포인트/트렌드는 기존과 동일 |
| 2026-09-15 | Claude | Grad-CAM으로 판단 근거 시각화 — `src/deep_learning/dl2_resnet/gradcam.py`(`layer4` 활성화·그래디언트 기반), data/test에서 클래스당 4장씩 원본+히트맵 그리드 저장 | - | - | 정성 분석(정확도 지표 아님). **배경이 아니라 물체 본체·손잡이에 정확히 집중** — 배경을 몰래 학습한 징후 없음. taper_step·mug 둘 다 손잡이 영역에 강하게 반응(둘 다 손잡이 있는 디자인이 많아 학습된 유효한 단서로 보임). 오분류 사례에서 원인 짐작 가능: straight가 mug로 오분류된 건(0.96 확신) 주목 영역이 몸통 중간 가로 띠에만 집중돼 전체 높이 비율을 못 본 것으로 보임 — 확신도 자체가 낮았던 오분류(taper_smooth→straight, 0.61)는 모델도 헷갈렸다는 신호로 해석 가능(주의가 입구·아래쪽 두 군데로 분산). 그래프: `reports/figures/resnet18/gradcam.png` |

| 2026-09-16 | Claude | 사용자가 `data/test1_orientation_backup`에서 마음에 안 드는 사진 32장을 직접 골라 제거(126장 남음) 후, 테스트1(Val)/테스트2(이 126장) 재평가 요청 | **테스트1: 96.9%, Macro-F1 0.968**(변화 없음, Val은 preprocess 기준이라 무관) | **테스트2: 82.5%(126장 중), Macro-F1 0.829** | `data/test1`(158장, 정리 전 원본)와 결과가 오차 범위 내(82.3%/0.818)로 일치 — 사용자가 걸러낸 32장이 특별히 쉽거나 어려운 쪽으로 쏠려있지 않았다는 뜻. 클래스별: straight 82.1%, taper_smooth 68.8%(소폭 개선), taper_step 97.0%, mug 81.8%(mug precision 100%로 개선) |
| 2026-09-16 | Claude | **최초 실험(백본 freeze) 환경 재현 검증** — "그때는 왜 낮았나"를 실측으로 확인하기 위해 최초 설정(freeze_backbone=True, 나머지 동일) 그대로 재학습. 공식 체크포인트 보호를 위해 `checkpoints_experiment/`, `reports/figures_experiment/`에 별도 저장(공식 자산 훼손 없음) | 88.2%(best epoch 13) | **63.5%(data/test1_orientation_backup 126장), Macro-F1 0.608** | 당시 기록(Val 91.3%, Test 58.9%, 56장 기준, 체크포인트 유실됨)과 비슷한 범위로 재현됨 — "백본을 얼려서 낮았다"는 설명이 실측으로 뒷받침됨. 현재 공식(unfreeze+TTA, Test 82.5%, Macro-F1 0.829)과 비교하면 백본 unfreeze만으로 거의 20%p 차이가 난다는 걸 재확인 |
| 2026-09-16 | Claude | **"테스트1/테스트2" 프레이밍 정리**(발표자료용) — 테스트1 = 같은 도메인 분할(`data/preprocess`의 Val, 크롤링 이미지 내 15~20% 홀드아웃), 테스트2 = 직접 촬영(`data/test1`, TTA 적용). `train.py`에 Val Macro-F1 출력도 추가(Test 쪽에만 있던 걸 통일) | **테스트1: 96.9%, Macro-F1 0.968** | **테스트2: 82.3%(130/158), Macro-F1 0.818** | 하락폭 14.6%p가 이 프로젝트가 검증하려는 domain shift 강건성 그 자체 — "모델이 원래 잘하는지"가 아니라 "크롤링 데이터로 배운 게 실사용 환경에도 통하는지"를 테스트1 vs 테스트2 격차로 한눈에 보여줄 수 있음 |
| 2026-09-16 | Claude | ResNet-18에 Test-Time Augmentation 추가(`--tta`, `train.py`) — 스케일 3종(리사이즈 232/256/280, 각 224 center-crop) x 좌우반전 = 6개 뷰의 softmax를 평균. 재학습 없이 기존 체크포인트로 `data/test1` 재평가 | 96.9%(동일 체크포인트) | 81.0%->**81.6%(129/158)**, Macro-F1 0.803->**0.813** | 재학습 없이 공짜로 얻은 개선. 클래스별: straight 82.0%(86.0%에서 소폭 하락), taper_smooth 65.9%(61.0%에서 개선, 4트랙 공통 최약체 클래스가 그나마 나아짐), taper_step 97.5%(동일), mug 81.5%(77.8%에서 개선). trade-off 있지만(straight 소폭 하락) 전체는 순개선 — `--eval-only --tta`로 재현 가능, 비용이 추론 시간 6배뿐이라 실사용에도 적용할 만함 |
| 2026-09-16 | Claude | ResNet-50(`--no-freeze-backbone`, ResNet-18과 동일 설정)으로 18 vs 50 첫 비교. `data/test1`(158장)로 평가 | **99.2%**(best epoch 6/11) | **81.0%(128/158)**, Macro-F1 **0.809** | ResNet-18(data/test1 기준 81.0%/0.803)과 **정확도는 동일**, Macro-F1은 근소 우위(0.809 vs 0.803). 다만 Val→Test 하락폭이 18.2%p로 ResNet-18(15.8%p)보다 큼 — Val 99.2%까지 거의 완벽히 맞춰서(train_acc도 98~99%대) 오히려 이 작은 데이터(515장)에 더 과적합했다는 신호로 보임. 클래스별: straight 80.0%, taper_smooth 58.5%(4트랙 공통으로 taper_smooth가 계속 가장 약함), taper_step 100%, mug 88.9%. **결론**: 이 데이터 규모에서는 50이 18 대비 뚜렷한 우위가 없음 — 파라미터 수만 늘려서 얻는 이득이 과적합 위험과 상쇄되는 것으로 보임, 굳이 50을 쓸 이유 약함 |
| 2026-09-17 | Claude | ResNet-18에 class-weighted loss 추가(`--class-weighted`) — 클래스별 표본 수 역수로 가중치(straight 0.913, taper_smooth 1.082, taper_step 1.11, mug 0.926)를 준 `CrossEntropyLoss`. `--no-freeze-backbone --tta`와 함께 `data/test1_orientation_backup`(126장)로 재학습·평가 | 96.9% | **84.1%, Macro-F1 0.834** (가중치 없는 TTA 버전 82.5%/0.829 대비) | 목표했던 taper_smooth는 F1 0.688→**0.716**로 개선됐지만, mug는 오히려 0.900→0.811로 하락(recall 81.8%→68.2%) — 표본 적은 클래스(taper_step/taper_smooth) 위주로 가중치를 준 결과 상대적으로 표본 많은 mug 비중이 줄어든 트레이드오프. 전체 지표는 순개선(정확도 +1.6%p, Macro-F1 +0.005)이지만 "만능 개선"은 아님. **공식 체크포인트는 아직 안 건드림** — `checkpoints_experiment/resnet18_shape.pth`에 별도 저장, 승격 여부는 미결정 |

## EfficientNet-B0 (`src/deep_learning/efficientnet/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-10-02 | Claude | 팀원(seungjae) `seungjae/efficientnet-b0-v2` 브랜치(`src/models/efficientnet/`, 데이터 경로 `img/train`·`img/test`)에서 로직 이식 — 이 프로젝트의 `data/preprocess` stratified_split(seed=42, dl2_resnet/vit와 동일 알고리즘)과 Test1/Test2 평가 구조에 맞춰 재작성. compound scaling + MBConv 구조, AdamW(lr=3e-4, wd=1e-4) + Cosine annealing, 15 epoch, `--no-freeze-backbone` | **96.9%, Macro-F1 0.969** | **88.1%(126장 중), Macro-F1 0.879** | 원본 브랜치가 PPT에 보고한 수치(Accuracy 0.88/Macro-F1 0.89, 단 `img/test` 126장 기준)와 오차범위 내로 일치 — 포팅이 정확하다는 게 검증됨. 4트랙 중 룰베이스·Mask R-CNN보다 훨씬 높고 ResNet-18(TTA, 82.5%/0.829)보다도 소폭 높음 |

---

## ConvNeXt-Tiny (`src/deep_learning/convnext/`)

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-10-02 | Claude | 팀원(이태인 추정) `convnext` 브랜치(주피터 노트북 `convnext_v3~v5.ipynb`, 체크포인트·원본 이미지가 그대로 커밋돼 있던 상태)에서 로직만 뽑아 `.py` 스크립트로 재작성. 2단계 학습(Stage1 warmup: head만 학습 5 epoch → Stage2 finetune: 전체 unfreeze+LLRD decay=0.8, 최대 40 epoch), 증강 RandomPerspective+RandomErasing(PPT "v3→v4" 단계와 동일), label_smoothing=0.1, patience=10 | **98.4%, Macro-F1 0.984** (epoch 19에서 조기 종료) | **86.5%(126장 중), Macro-F1 0.867** | **주의**: PPT는 마지막 단계를 "val loss 기준 early stopping"이라고 설명하지만 실제 `convnext_v5.ipynb` 코드는 이와 다르게 validation을 아예 없애고 train_acc 기준으로 멈추며 held-out 데이터를 test셋에 합쳐버리는 방식이었음(PPT 서술과 노트북 코드 불일치 발견). 이 스크립트는 노트북을 그대로 베끼지 않고 **PPT가 설명한 대로** val loss 기준 best-model 선택을 실제로 구현함 — val을 끝까지 held-out으로 유지하는 게 더 안전한 관행이라 판단. 그 결과 PPT가 보고한 수치(0.88~0.89)보다 다소 낮게 나왔는데(0.867), 이는 더 엄격한 평가 방식(검증 데이터 미누출) 때문으로 추정 — 원인 미확정 |

---

## Custom CNN (`src/deep_learning/custom_cnn/`) — 구현 공부/실험용, 4트랙 비교 제외

**4트랙(룰베이스/Mask R-CNN/ResNet/EfficientNet) 공식 비교표에는 넣지 않는다.** 사전학습 없이 직접 설계한 작은 CNN(conv block 4개+GAP+Dropout+FC, `model.py`)을 `data/preprocess`만으로 처음부터 학습해서, "왜 전이학습(ResNet)이 유리한가"를 확인하는 대조군 실험.

| 날짜 | 담당자 | 변경 사항 | Val 정확도 | Test 정확도 | 비고 |
|---|---|---|---|---|---|
| 2026-09-16 | Claude | `model.py`(SimpleCNN, 채널 32→64→128→256) + `train.py`(dl2_resnet과 동일 패턴, 입력 128x128, RandomPerspective 포함) 최초 구현, 40 epoch 학습 | 66.9%(best epoch 33) | **24.0%(25/104), Macro-F1 0.233** | Val 곡선이 처음부터 끝까지 진동함(예: epoch 28에서 val_acc 0.480→0.339로 급락 후 재상승) — 사전학습 특징 없이 515장만으로 학습하니 일반화가 불안정하다는 신호. **Test에서 완전히 무너짐**: straight recall 7.9%, taper_smooth recall 9.1% — 대부분 mug로 예측(mug recall 100%, precision 12.0%)해서 룰베이스와 비슷한 "애매하면 mug" 붕괴 패턴 재현. **결론**: ResNet-18 unfreeze(Test 79.8%, Macro-F1 0.769)와의 격차(정확도 55.8%p, Macro-F1 0.536)가 이 데이터 규모(train 515장)에서 전이학습이 얼마나 결정적인지 정량적으로 보여줌 — 처음부터 학습한 CNN은 ImageNet에서 배운 저수준 시각 특징(에지·질감·색 대비) 없이는 이 정도 소규모 데이터로 일반화하기 어려움 |

---

## Vision Transformer (`src/deep_learning/vit/`) — 팀원(taehyun) 브랜치 비교용, 4트랙 비교 제외

**4트랙(룰베이스/Mask R-CNN/ResNet/EfficientNet) 공식 비교표에는 넣지 않는다.** 팀원 taehyun님의 `taehyun/ViT` 브랜치(DeiT-Small/16, timm) 로직을 이식해서, 이 프로젝트의 `data/preprocess`(642장)로 재학습·비교한 것. taehyun님 원본 브랜치는 별도 수집한 441장(우리 `data/raw2`와 동일 원본 이미지 pool에서 다르게 추린 것, sha256 대조로 확인)으로 학습했음 — 여기서는 "같은 데이터로 학습했을 때" 비교를 위해 `data/preprocess` 기준으로 재현.

`--baseline`(완전 베이스라인: 증강·종횡비패딩·부분freeze·차등LR·weight decay 전부 제거, 백본 전체를 단일 LR로 학습)과 `--experiment B`(taehyun 설계: 마지막 4블록만 unfreeze+차등LR+weight decay 0.01+종횡비 패딩+증강) 두 버전을 각각 테스트1(`data/preprocess`의 Val, 127장)/테스트2(`data/test1_orientation_backup`, 126장) 기준으로 비교.

| 날짜 | 담당자 | 변경 사항 | 테스트1(Val) | 테스트2 | 비고 |
|---|---|---|---|---|---|
| 2026-09-17 | Claude | `model.py`/`transforms.py`/`train.py` 이식(timm `deit_small_patch16_224.fb_in1k`). `--baseline`(완전 베이스라인, 백본 전체 unfreeze+단일LR+증강없음+단순 Resize/CenterCrop) 25 epoch 학습 | **98.4%, Macro-F1 0.984** | **85.7%(108/126), Macro-F1 0.856** | 하락폭 12.7%p. ResNet-18(TTA, 82.5%/0.829)보다도 소폭 높음 — 놀랍게도 "아무 튜닝도 안 한" 버전이 더 잘 나옴 |
| 2026-09-17 | Claude | 동일 데이터로 taehyun 설계(`--experiment B`: 마지막 4블록+norm+head만 unfreeze, 차등LR 1e-5/1e-4, weight_decay=0.01, 종횡비 보존 패딩, 좌우반전+회전+ColorJitter 증강) 25 epoch 학습, 같은 테스트셋으로 재평가(`--eval-only`) | 99.2%, Macro-F1 0.992 | **73.8%(93/126), Macro-F1 0.729** | 하락폭 25.4%p — **완전 베이스라인보다 오히려 낮음**(테스트2 정확도 -11.9%p, Macro-F1 -0.127). 두 버전 다 테스트1(Val)은 거의 완벽(98~99%)인데 taehyun 설계 쪽이 실사진에서 더 크게 무너짐. 추정 원인: 마지막 4블록만 풀고 나머지 8블록을 ImageNet 가중치로 고정해둔 게, 오히려 우리 도메인(실사진 원근왜곡 등)에 적응할 여지를 줄여서 일반적 직관(부분 고정=과적합 방지)과 반대 결과가 나온 것으로 추정 — 원인 미확정, 추가 검증 필요 |

---

## 4갈래 종합 비교

`notebooks/03_model_comparison.ipynb`에서 정기적으로 산출되는 비교 결과를 요약한다. Train(웹 이미지)→Test(실사용 이미지) 간 domain shift로 인한 정확도 하락은 의도된 평가이니, 하락 폭 자체를 비교 지표로 다룬다.

| 날짜 | 트랙 | Val→Test 정확도 하락폭 | 주요 혼동 클래스 쌍 | 비고 |
|---|---|---|---|---|
| 2026-09-15 | 룰베이스 | raw2(≈val 성격) 65.8% → **Test 18%** (47.7%p 하락) | straight/taper_smooth/taper_step 전부 → mug로 대량 오분류 | `notebooks/test_set_evaluation.ipynb`. data/test는 직접 촬영한 진짜 Test 세트(56장). 원인: (1) 위에서 내려다본 촬영 각도 때문에 원근 왜곡으로 직선이 테이퍼져 보이고 키가 눌려 보임 — README가 이미 예견한 위험(\"각도에 따라 왜곡/소실\"), (2) 일부 taper_step은 원래도 짧고 통통한 디자인이라 mug 경계(h/d≤1.5)에 가까워서 살짝만 눌려도 넘어감. 코드 버그 아님 — 2D 폭 프로파일 방식의 근본 한계, 촬영 각도(정면 비율)를 프로토콜대로 지키거나 원근 보정이 필요 |
| 2026-09-15 | Mask R-CNN | raw2 88.0% → **Test 39%** (49.0%p 하락) | straight → taper_step/taper_smooth로 분산, mug → taper_step 대량 오분류 | 같은 원인(촬영 각도). Mask R-CNN의 마스크 자체는 룰베이스보다 낫지만, 마스크 이후의 `classify_shape()`(폭 프로파일 기반)는 두 트랙이 공유하므로 각도 왜곡에는 똑같이 취약함 — 마스크 품질 문제가 아니라 기하학적 가정(정면 촬영)이 깨진 것 |
| 2026-09-15 | ResNet-18 | preprocess val 91.3% → **Test 58.9%** (32.4%p 하락) | mug → taper_step 대량 오분류(6장 중 4장, 표본 작음) | **같은 domain shift인데도 하락 후 절대 정확도가 룰베이스(18%)·Mask R-CNN(39%)보다 훨씬 높음.** 기하 규칙 대신 학습된 시각 패턴을 쓰는 게 촬영 각도 왜곡에 강하다는 가설 확인. 각도 왜곡을 겨냥한 `RandomPerspective` 증강 포함. 4트랙 중 처음으로 "회피"가 아니라 "정면 돌파"에 가까운 결과 |
| 2026-09-15 | (재측정) | `data/test`가 56장→104장으로 확대(test2 병합, mug 6→10장)된 뒤 같은 세트로 재측정 | - | Mask R-CNN 제로샷 39%→**46.2%**(48/104), ResNet-18(freeze) 58.9%→**64.4%**(67/104, 재학습 포함) | 표본이 커지니 두 트랙 다 정확도가 올라가고 순위는 그대로 유지(ResNet > Mask R-CNN 제로샷). mug 표본이 늘면서(6→10) 이전 "mug만 급락" 노이즈가 해소됨. **참고**: 룰베이스 pseudo-label로 fine-tuning한 Mask R-CNN 체크포인트도 같은 104장으로 평가해봤는데 **15.4%(16/104)로 제로샷의 1/3 수준** — confusion matrix상 거의 전부 mug로만 예측(class collapse). raw2 기준(85%→78%)보다 실제 Test에서 훨씬 크게 무너짐 — fine-tuning 재시도 안 하는 게 맞다는 결론 재확인 |
| 2026-09-15 | ResNet-18(백본 unfreeze) | preprocess val 96.9% → **Test 79.8%**(83/104) (17.0%p 하락) | 상대적으로 taper_smooth가 제일 약함(13/22, 59%), 나머지는 76~94% | **현재까지 전 실험 통틀어 최고 기록이자 최소 하락폭.** 백본을 프리즈한 버전(Val 89.8%→Test 64.4%, 25.3%p 하락) 대비 절대 정확도·하락폭 둘 다 개선 — "학습된 시각 패턴이 각도 왜곡에 강하다"는 가설이 백본까지 풀었을 때 더 강하게 확인됨. 학습 시간은 프리즈 버전과 거의 동일(145s, 이 데이터 규모에서는 역전파 비용이 병목 아님) — 시간 비용 없이 큰 이득. 이후 dl2_resnet 실험 기본값으로 채택 |
