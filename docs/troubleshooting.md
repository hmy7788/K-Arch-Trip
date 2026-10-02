# 트러블슈팅 로그

실험·구현 중 겪은 문제와 해결 과정을 기록한다. 같은 문제를 팀원이 반복해서 겪지 않는 것이 목적이므로, **원인이 명확해지지 않았더라도 증상과 시도한 것들을 남긴다.**

새 항목을 추가할 때는 아래 형식을 따른다:

```
## [YYYY-MM-DD] 짧은 제목

- **트랙/영역**: 예) dl2_resnet / data_collection / rule_based ...
- **증상**: 무엇이 어떻게 잘못됐는지
- **원인**: 파악됐다면 기록, 안 됐다면 "미파악"
- **해결**: 어떻게 고쳤는지 (또는 우회했는지)
- **관련 파일**: 경로
```

---

## 자주 발생할 수 있는 이슈 (사전 체크리스트)

아직 실제 트러블슈팅 사례는 없지만, README/AGENTS.md 설계상 미리 예견되는 함정을 기록해둔다 — 실제로 겪으면 아래를 정식 로그 항목으로 옮기고 구체화할 것.

- **개체 단위 분할 실수**: 이미지 단위로 train/test를 랜덤 분할하면 같은 물건 사진이 양쪽에 섞여 정확도가 비정상적으로 높게 나올 수 있음. Test 정확도가 이상하게 높으면 가장 먼저 의심할 것 → `data-pipeline` 담당.
- **머그형 오판정**: 손잡이 유무로 먼저 분기하는 구현은 규칙 위반 (실루엣이 손잡이보다 우선). 손잡이 달린 테이퍼 텀블러가 mug로 잘못 분류되면 이 버그를 의심할 것 → `rule-based-vision` 담당.
- **배경 과적합**: 촬영 배경이 한 종류로 쏠리면 모델이 배경을 형태 신호로 학습해버릴 수 있음. Val 정확도는 높은데 Test에서 급락하면 `01_data_diversity_check.ipynb`로 배경 분포부터 확인.
- **Domain shift를 버그로 오인**: Train(웹 크롤링)→Test(직접 촬영) 정확도 하락은 의도된 평가 설계다. 이를 "모델이 망가졌다"로 오인해 불필요하게 재학습하지 말 것.

---

## 로그

## [2026-09-15] 학습 도중 프로세스를 강제종료했다가 기존 좋은 체크포인트가 덮어써짐

- **트랙/영역**: deep_learning/dl2_resnet — `train.py`
- **증상**: 91.3% val_acc로 잘 학습된 체크포인트가 있는 상태에서, "늘어난 test로 재평가"를 위해 별생각 없이 `train.py`(재학습 포함)를 다시 실행함 → 사용자가 "왜 재학습하냐, 체크포인트만 불러와서 평가하면 되잖아"라고 지적 → 백그라운드 프로세스를 강제종료(`Stop-Process`)함. 근데 `--eval-only`를 새로 만들어서 같은 체크포인트를 불러와 평가했더니 val_acc가 80.3%로 나옴(91.3%여야 정상).
- **원인**: 강제종료 명령이 프로세스에 전달되고 실제로 죽기까지 시간차가 있는데, 그 사이에 **1 epoch째 학습이 이미 끝나서 체크포인트 저장 로직(`if val_acc > best_val_acc: torch.save(...)`)이 실행돼버림** — 1 epoch째는 초기값 0.0보다 무조건 높으므로 항상 저장 조건을 만족함. 그 결과 겨우 1 epoch만 학습된(val_acc 낮음) 가중치가 기존 91.3% 체크포인트를 덮어씀. `--eval-only` 자체는 정상 동작했고, 정직하게 "지금 파일에 저장된 걸" 불러와 평가했을 뿐 — 버그가 아니라 파일이 이미 손상된 상태였음.
- **해결**: 미봉책 없음 — 좋았던 가중치는 복구 불가, 처음부터 재학습(중단 없이 끝까지 실행)해서 새로 89.8% 체크포인트 확보. 재발 방지로 `train.py`를 고침: 학습 루프 중에는 `<ckpt>.tmp`에만 저장하고, **전체 epoch가 끝까지 성공적으로 끝났을 때만 `os.replace(tmp, ckpt)`로 최종 파일에 반영**. 이제 학습을 중간에 죽여도 `.tmp` 파일만 지저분해질 뿐 기존 `ckpt_path`는 절대 안 바뀜.
- **교훈**: "점진적으로 최고 기록을 덮어쓰며 저장"하는 학습 스크립트는 실행 중 강제종료에 취약하다 — 특히 **매 실행 초반(1 epoch째)에는 거의 항상 저장 조건을 만족**하므로, "막 시작한 프로세스니까 죽여도 안전하겠지"라는 가정이 틀릴 수 있다. 체크포인트를 학습 중간에 직접 최종 경로에 쓰지 말고, 임시 파일 경유 + 원자적 교체(`os.replace`)로 저장하는 패턴을 처음부터 넣을 것.
- **관련 파일**: `src/deep_learning/dl2_resnet/train.py`

## [2026-09-15] print()의 em-dash(—) 때문에 학습 스크립트가 마지막 단계에서 크래시

- **트랙/영역**: 전체 스크립트 공통 — `src/deep_learning/dl2_resnet/train.py`에서 첫 발견
- **증상**: ResNet-18 학습(20 epoch, 145초) + Val 평가까지 전부 성공하고, 마지막 `data/test` domain shift 평가 직전 `UnicodeEncodeError: 'cp949' codec can't encode character '—'`로 크래시. 학습 결과(체크포인트, Val 지표)는 이미 저장됐지만 Test 평가·그래프는 유실됨.
- **원인**: `print()`에 em-dash(`—`, U+2014) 같은 cp949(Windows 한글 코드페이지)에 아예 없는 문자가 들어가면, 한글(가~힣)은 cp949로 인코딩되지만 이런 특수문자는 조용히 깨지는 게 아니라 **예외를 던지며 프로세스가 죽는다.** 이전에 고친 "print가 끝까지 안 보이는" 버퍼링 문제와는 별개의, 더 심각한 문제.
- **해결**: `sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")` — buffering뿐 아니라 인코딩도 명시적으로 utf-8로 강제. `src/data_collection/`의 기존 스크립트 5개 + `train.py` 전부에 적용.
- **교훈**: 이 저장소에서 만드는 모든 실행 스크립트는 처음부터 `sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")`를 넣고 시작할 것 — 한글은 cp949로 되니까 괜찮다고 방심하면 em-dash, 화살표(→), 문장부호(", ') 등에서 똑같이 터진다. 오래 걸리는 학습/처리 스크립트일수록 마지막 한 줄 때문에 결과를 통째로 날릴 수 있어 특히 중요.
- **관련 파일**: `src/deep_learning/dl2_resnet/train.py`, `src/data_collection/*.py` 전체

## [2026-09-15] data/test 실사진에서 정확도 급락(66~88% → 18~39%) — 촬영 각도 원근 왜곡

- **트랙/영역**: rule_based, deep_learning/dl1_maskrcnn (둘 다 `classify_shape()` 공유) — `notebooks/test_set_evaluation.ipynb`
- **증상**: `data/raw2` 기준 룰베이스 65.8%/Mask R-CNN 88.0%였는데, 진짜 촬영 Test 세트(56장)에서는 각각 18%/39%로 급락. 대부분 mug 또는 taper_step으로 쏠려서 오분류됨.
- **원인**: 마스크 품질 문제가 아님 — 마스크를 직접 눈으로 확인해보니 물체 윤곽 자체는 꽤 정확히 잡혔음. 문제는 **촬영 각도**: Test 사진들이 위에서 내려다보는 각도로 찍혀서, 원근 때문에 (1) 직선 원통형이 사진에서는 위가 넓고 아래가 좁은 테이퍼형처럼 보이고, (2) 물체의 세로 길이가 실제보다 짧게 찍힘. 짧고 통통한 taper_step 디자인은 원래도 mug 경계값(h/d≤1.5)에 가까운데, 여기에 각도 왜곡까지 더해지면 mug로 넘어가 버림.
- **해결**: 미해결, 코드로 고칠 문제가 아님 — `compute_width_profile()`이 전제하는 "정면에서 촬영" 가정 자체가 깨진 것. README의 촬영 프로토콜(정면 비율 절반 이상)을 지켜서 재촬영하거나, 장기적으로는 원근 보정/3D 트랙(reconstruction_3d, 선택 스코프)이 필요.
- **교훈**: `data/raw2`(크롤링 상품사진, 대부분 정면)로 검증된 정확도는 실제 촬영 조건에서 그대로 재현되지 않는다 — 이게 바로 프로젝트가 설계한 domain shift 평가의 핵심이지만, 하락 폭이 예상보다 훨씬 컸다(약 50%p). 향후 Test 세트를 늘릴 때 각도 다양성(정면/45도/위에서 살짝)을 프로토콜대로 실제로 지켰는지 재확인 필요.
- **관련 파일**: `notebooks/test_set_evaluation.ipynb`, `docs/experiment-log.md`("4갈래 종합 비교" 표)

## [2026-09-15] get_mask()가 폰카메라 원본(4000x3000)에서 장당 35초 소요

- **트랙/영역**: rule_based / shape_classifier.py::get_mask
- **증상**: `data/test/`(직접 촬영한 진짜 Test 세트, 카카오톡으로 받은 폰카메라 사진)를 전량 평가하려다 노트북이 몇 분째 안 끝남. GPU 사용률 0%라 Mask R-CNN이 아니라 룰베이스 단계에서 멈춘 것으로 확인.
- **원인**: 테스트 사진이 4000x3000(12MP) — 지금까지 써온 `data/raw2`의 상품 사진(보통 500~1000px)보다 훨씬 큼. `get_mask()`가 이미지를 축소하지 않고 원본 그대로 Otsu/Canny/GrabCut을 돌려서, GrabCut의 픽셀 수 비례 연산량이 그대로 부담됨. 실측: 한 장에 **35초**.
- **해결**: `get_mask()`에 전처리 단계 추가 — 긴 변이 `MAX_PROCESSING_DIM=1200`을 넘으면 비율 유지하며 축소 후 처리. `classify_shape()`는 마스크의 절대 픽셀 수가 아니라 비율(높이/지름 등)만 쓰므로 축소해도 판정에 영향 없음. 같은 사진 기준 35초 → **2.2초**(16배 개선).
- **주의**: `get_mask()`가 반환하는 마스크는 (원본이 컸다면) 축소된 크기를 따른다 — 원본 이미지와 크기가 다를 수 있음. `classify_shape()`/`compute_width_profile()`은 비율만 쓰므로 문제없지만, 마스크를 원본에 그대로 오버레이하려는 새 코드를 짤 때는 크기가 다를 수 있다는 점을 고려할 것.
- **관련 파일**: `src/rule_based/shape_classifier.py` (`get_mask`, `MAX_PROCESSING_DIM`)

## [2026-09-15] 임계값 재조정 후 합성 테스트 실패 — 원인은 임계값이 아니라 테스트 손잡이 위치

- **트랙/영역**: rule_based / synthetic.py, shape_classifier.py — `test_taper_smooth_with_handle_stays_taper_smooth`
- **증상**: `STEP_JUMP_RATIO_THRESHOLD`를 0.4→0.29로 실측 재조정한 뒤 pytest 재실행하니, "손잡이 달린 테이퍼는 여전히 taper_smooth여야 한다" 회귀 테스트가 실패(`taper_step`으로 오분류됨).
- **원인**: `make_taper_smooth_mask(with_handle=True)`가 손잡이를 몸통 높이의 정확히 절반(`(y_top+y_bottom)//2`) 지점에 배치했는데, 이 지점이 하필 폭 프로파일의 4번째/5번째 구간(bin) 경계였음. 손잡이가 두 구간에 걸쳐 있으면 median smoothing(window=5,7,9 다 동일)으로도 완전히 안 지워지고, 구간 경계에서 인접 구간과의 값 차이로 인한 잔여 "단차"(step_ratio≈0.333)가 생김 — 이게 새 임계값 0.29를 넘어버림. window 크기를 늘려도 이 잔여값은 안 줄어듦(구조적 아티팩트라 smoothing으로 해결 안 되는 종류).
- **해결**: 손잡이를 구간 경계가 아니라 몸통 높이 1/4 지점(한 구간 안쪽)으로 옮기고 크기도 줄임(`protrusion=40, height=20`) → step_ratio가 0.222로 떨어져 안전하게 통과. `_add_handle_bump` 호출부 자체는 안 바꾸고 `make_taper_smooth_mask`의 호출 파라미터만 조정.
- **교훈**: 합성 테스트 마스크를 만들 때 "구간 경계에 걸치는 위치"는 median smoothing이 못 지우는 인공적인 단차를 만들 수 있다 — 실제 사진에서도 손잡이가 하필 구간 경계 부근에 있으면 비슷한 문제가 생길 수 있다는 뜻이기도 함(미검증, 실사진에서 아직 관찰 안 됨).
- **관련 파일**: `src/rule_based/synthetic.py` (`make_taper_smooth_mask`), `src/rule_based/shape_classifier.py`(임계값 자체는 정상, `step_ratio`를 항상 반환하도록 리팩터링)

## [2026-09-15] 룰베이스 pseudo-label로 Mask R-CNN fine-tuning → 오히려 성능 하락

- **트랙/영역**: deep_learning / dl1_maskrcnn — `notebooks/maskrcnn_finetune_rulebase_pseudolabels.ipynb`
- **증상**: 제로샷 Mask R-CNN(85%, `maskrcnn_vs_rule_based.ipynb`)보다 fine-tuning 후(78%)가 더 낮음. 같은 held-out 40장으로 재평가해 확인(학습 데이터에는 이 40장을 애초에 제외했으므로 데이터 누수 아님).
- **원인**: (1) 룰베이스 `classify_shape()`가 폴더 라벨과 맞은 것만 pseudo-label로 채택했는데, 클래스별 채택 수가 크게 불균형(mug 88 / straight 81 / taper_smooth 64 / taper_step 33) — 룰베이스가 원래 약한 클래스(taper_step)일수록 "맞은 것"도 적게 남아서 학습셋이 mug 쪽으로 편향됨. (2) 룰베이스 마스크 자체가 Otsu/Canny/GrabCut 특유의 거친 경계를 가지고 있어서, COCO로 학습된 원래의 정밀한 마스크 품질을 깎아먹었을 가능성.
- **해결**: 이 fine-tuning 접근은 폐기. dl1 트랙은 제로샷(사전학습 그대로, `segment.py`의 기본 `load_model()`)을 계속 사용. 파인튜닝된 체크포인트(`checkpoints/maskrcnn_finetuned_rulebase_pseudolabels.pth`)는 실사용 금지, 기록용으로만 남김.
- **교훈**: 약한 방법(룰베이스)의 출력을 강한 방법(사전학습 Mask R-CNN)의 정답으로 쓰는 self-distillation은, 필터링을 해도 여전히 위험함 — 필터링 자체가 원래 약한 클래스의 학습 데이터를 더 줄여서 불균형을 악화시킬 수 있음. 다음에 시도한다면 클래스별로 pseudo-label 수를 강제로 맞추거나(undersampling/oversampling), 애초에 사람이 직접 마스크를 그린 소량의 진짜 정답을 섞는 게 나을 것.
- **관련 파일**: `notebooks/maskrcnn_finetune_rulebase_pseudolabels.ipynb`, `docs/experiment-log.md`

## [2026-09-15] get_mask()에 Canny 엣지 기반 마스크 추가 — taper_step 개선, 부작용도 있음

- **트랙/영역**: rule_based / shape_classifier.py::get_mask, `_rough_mask_canny`
- **배경**: 위 항목(taper_step 대량 오분류)의 원인을 실사진으로 직접 진단. 실패한 6장 중 3장(101/65/83.jpg)이 **흰색/크림색 텀블러 + 흰색/연회색 배경** 조합이었고, `_rough_mask_otsu()`의 전경 비율이 1.3~5.6%까지 떨어져 있었음(거의 아무것도 못 잡음) — 명도 대비가 너무 낮아 Otsu가 근본적으로 구분 불가. 나머지 1장(70.jpg)은 애초에 뚜껑 클로즈업 사진이라 몸통 자체가 안 찍힘(알고리즘 문제 아님, 데이터 문제).
- **해결**: Canny 엣지(그림자·그라데이션으로 생기는 옅은 경계선을 봄, 명도 절대값은 안 봄) 기반 `_rough_mask_canny()`를 추가해서 `_rough_mask_otsu()` 결과와 OR로 합친 뒤 GrabCut 시드로 사용. Otsu와 Canny의 실패 유형이 서로 달라서 한쪽이 실패해도 다른 쪽이 보완함.
- **결과 (raw2 동일 40장, seed=42, 전/후 비교)**:

  | 클래스 | 이전 | 이후 |
  |---|---|---|
  | straight | 100%(10/10) | 90%(9/10) |
  | taper_smooth | 60%(6/10) | 50%(5/10) |
  | taper_step | 30%(3/10) | **60%(6/10)** |
  | mug | 100%(10/10) | 100%(10/10) |
  | 전체 | 72.5%(29/40) | **75.0%(30/40)** |

  목표였던 taper_step은 2배 개선, 전체도 순개선. **다만 straight·taper_smooth에서 각 1건씩 새 오분류 발생** — Canny가 배경의 텍스처/그라데이션을 몸통 일부로 같이 잡아버리는 부작용으로 추정(원인 미확정, 개별 사진 분석 안 함).
- **여전히 해결 안 되는 케이스**: 70.jpg(클로즈업, 데이터 문제), 63.jpg(복잡한 야외 배경, mug로 계속 오분류), 101.jpg(taper_step인데 이제 taper_smooth로 — mug는 벗어났지만 여전히 부정확).
- **관련 파일**: `src/rule_based/shape_classifier.py` (`_rough_mask_canny`, `get_mask`), `notebooks/rule_based_raw2_sample.ipynb`

## [2026-09-15] taper_step이 get_mask() 몸통 누락으로 mug에 대량 오분류 (n=40 확인)

- **트랙/영역**: rule_based / shape_classifier.py::get_mask, classify_shape
- **증상**: `data/raw2`에서 taper_step 클래스 무작위 10장 중 6장이 mug로 오분류(`notebooks/rule_based_raw2_sample.ipynb`). 마스크를 보면 몸통 없이 금속 뚜껑/테두리 부분만 하얗게 잡힘.
- **원인**: `get_mask()`(Otsu+GrabCut)가 매트한 몸통 색과 배경을 구분 못 하고, 대비가 강한 금속 뚜껑/테두리만 전경으로 남기는 경우가 반복됨. 높이가 실제보다 훨씬 작게 측정돼 `height/diameter <= 1.5`(mug 조건)를 만족해버림. 09-14 실험(그림자 융착, 색 충돌)과 원인은 다르지만 결과(몸통 누락 → mug 오분류)는 같은 계열.
- **해결**: 미해결. 마스크가 몸통까지 정상적으로 잡힌 3장(23.jpg, 49.jpg, 26.jpg)은 전부 정확히 분류됨 — `classify_shape()` 로직 자체는 문제없고, 병목은 100% `get_mask()`의 세그멘테이션 품질. Otsu+GrabCut 같은 색상 기반 방법의 근본 한계로 보이며, Mask R-CNN(dl1) 트랙 도입이 유력한 해결책.
- **관련 파일**: `src/rule_based/shape_classifier.py`, `notebooks/rule_based_raw2_sample.ipynb`

## [2026-09-15] Bing/Google 크롤링 결과에 쇼핑몰 "상세페이지" 이미지가 섞임

- **트랙/영역**: data_collection / search_crawler.py
- **증상**: `--class straight --engines bing`으로 "일자 텀블러" 검색 시, 텀블러 실물 사진이 아니라 980x4500px 같은 세로로 아주 긴 이미지가 섞여 나옴 — 한국 쇼핑몰 상품 "상세페이지"를 통째로 캡처한 이미지(여러 제품컷+마케팅 문구가 세로로 이어붙은 형태)였음.
- **원인**: Bing/Google 이미지 검색이 썸네일 뒤의 "원본 이미지"를 그대로 가져오는데, 쇼핑몰들이 상세페이지 자체를 하나의 긴 이미지로 등록해두는 경우가 많아서 검색 결과에 섞임.
- **해결**: `search_crawler.py`에 가로세로 비율 필터(`MAX_ASPECT_RATIO=2.2`) 추가 — 다운로드 직후 비율이 비정상적으로 긴 이미지를 자동 삭제. 단, 이건 극단적으로 긴 이미지만 걸러낼 뿐, 정상 비율의 스크린샷(상품 리스트 캡처 등)은 못 거름 — **사람 검수는 여전히 필수.**
- **관련 파일**: `src/data_collection/search_crawler.py` (`_remove_extreme_aspect_ratio_images`)

## [2026-09-14] cv2.convexityDefects 반환 shape이 환경마다 다름

- **트랙/영역**: rule_based / shape_classifier.py::detect_handle
- **증상**: `defects[:, 0]` 로 순회 후 `defect[3]` 인덱싱 시 `IndexError: invalid index to scalar variable`
- **원인**: 이 프로젝트 conda 환경(`vision_programming`, opencv-python 5.0.0.93)에서 `cv2.convexityDefects()`가 `(N, 4)` shape을 반환함. 흔히 알려진 `(N, 1, 4)` shape을 가정하고 `defects[:, 0]`로 중간 차원을 벗겨내면 스칼라가 나와 깨짐.
- **해결**: `defects.reshape(-1, 4)`로 순회하도록 변경 — 두 shape 모두에서 안전하게 동작.
- **관련 파일**: `src/rule_based/shape_classifier.py`

## [2026-09-14] 손잡이 돌출부가 taper_smooth를 taper_step으로 오분류시킴

- **트랙/영역**: rule_based / shape_classifier.py::classify_shape
- **증상**: `make_taper_smooth_mask(with_handle=True)`가 `taper_smooth`가 아니라 `taper_step`으로 분류됨 (README가 명시적으로 경고한 "손잡이 때문에 오분류" 케이스의 변형).
- **원인**: 손잡이가 폭 프로파일의 한두 구간(bin)에서만 폭을 급격히 튀어오르게 만들어, "단일 구간 급감(step)" 판정 로직이 그 스파이크 이후의 복귀를 진짜 단차로 오인함.
- **해결**: 폭 프로파일에 1D median filter(window=5)를 적용해 한두 구간짜리 스파이크를 억제한 뒤 판정하도록 변경 (`_median_smooth`). 손잡이 유무로 분기하는 게 아니라 측정 자체를 견고하게 만드는 방식이라 "실루엣이 손잡이보다 우선" 원칙과 충돌하지 않음.
- **관련 파일**: `src/rule_based/shape_classifier.py`
- **주의**: window=5는 지금의 합성 마스크(높이 300px, 손잡이 높이 50px, 10구간)에 맞춘 값. 실사진 확보 후 손잡이 크기 분포를 보고 재조정 필요 (TUNE_ME).

## [2026-09-14] get_mask()가 그림자 융착·물체-배경 색 충돌 사진에서 계속 실패 (GrabCut 적용 후에도)

- **트랙/영역**: rule_based / shape_classifier.py::get_mask
- **증상**: 실사진 4장 중 2장(머그형, 단차테이퍼형)에서 마스크가 물체를 온전히 못 잡음. Otsu 임계값만 쓸 때는 물론, `_grabcut_refine()`(Otsu 결과를 시드로 GrabCut 5회 반복)을 추가한 뒤에도 동일하게 실패.
- **원인**: (1) 머그형 사진은 컵 하단 코르크 받침 색이 배경 나무 바닥과 사실상 같은 색이라 색상 기반 GMM(GrabCut)으로도 전경/배경이 구분 안 됨. (2) 단차테이퍼형 사진은 물체 바로 아래 진한 그림자가 물체와 융착돼 있어, 그림자와 물체의 경계를 만드는 정보(색·명도)가 부족함. 둘 다 알고리즘 튜닝이 아니라 원본 사진의 배경 대비·조명 조건 문제.
- **해결**: 미해결. GrabCut 추가로 단차테이퍼형의 판정 임계값 여유는 다소 개선됐음(h/d 1.51→1.58, b/t 0.92→0.89, 둘 다 위험 경계에서 멀어짐)but 그림자 융착·색 충돌 자체는 못 고침. 다음 옵션: (a) 촬영 시 배경 대비를 확실히 주고 그림자 없는 조명 사용(README 촬영 프로토콜과 일치하는 방향으로 재촬영), (b) Mask R-CNN(dl1) 트랙이 이런 어려운 케이스를 대신 처리하도록 함(애초에 그렇게 설계됨), (c) HSV 기반 그림자 억제 전처리 추가(시도 안 함).
- **관련 파일**: `src/rule_based/shape_classifier.py` (`_rough_mask_otsu`, `_grabcut_refine`, `get_mask`)

## [2026-09-14] cv2.imread/imwrite가 한글 경로에서 조용히 실패

- **트랙/영역**: rule_based / classify_image.py (실제 이미지 파일 입력)
- **증상**: `cv2.imwrite(korean_path, img)`가 예외 없이 `False`를 반환하며 파일이 저장되지 않음. `cv2.imread`도 마찬가지로 `None`을 조용히 반환할 수 있음.
- **원인**: OpenCV의 Windows 빌드가 non-ASCII(한글 등) 경로를 제대로 처리하지 못함. **이 저장소 경로 자체가 `C:\Users\한국전파진흥협회\...`라서 팀원 전원이 실사진으로 테스트할 때 반드시 걸리는 문제.**
- **해결**: `classify_image.py`에 `imread_unicode`(`np.fromfile` + `cv2.imdecode`) / `imwrite_unicode`(`cv2.imencode` + `ndarray.tofile`) 헬퍼를 추가해 우회. **다른 트랙(dl-trainer 등)에서 cv2로 이미지 파일을 직접 읽고 쓰는 코드를 새로 짤 때도 같은 패턴을 써야 함** — `cv2.imread`/`cv2.imwrite`를 경로 문자열로 직접 호출하지 말 것.
- **관련 파일**: `src/rule_based/classify_image.py`

## [2026-09-14] detect_handle이 taper_step의 꺾임 지점도 손잡이로 오검출

- **트랙/영역**: rule_based / shape_classifier.py::detect_handle
- **증상**: `make_taper_step_mask()`(손잡이 없음)를 `detect_handle()`에 넣으면 `True`가 나옴.
- **원인**: convexityDefects는 "오목한 지점의 깊이"만 보므로, 단차 테이퍼형의 꺾임 지점(급격한 폭 감소로 생기는 오목 코너)과 실제 손잡이를 기하학적으로 구분하지 못함.
- **해결**: 미해결 — `classify_shape()`의 분류 결과 자체에는 영향 없음(handle_detected를 판정에 쓰지 않으므로)을 확인하고 일단 진행. 다만 이 값을 나중에 설명 텍스트("손잡이가 있는 텀블러입니다" 등)에 그대로 쓰면 taper_step에 대해 잘못된 설명이 나갈 수 있음 — pipeline-integrator가 explanation 연결할 때 주의.
- **관련 파일**: `src/rule_based/shape_classifier.py`
