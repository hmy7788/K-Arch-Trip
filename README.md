# K-건축 여행 도감

**텀블러 형태를 대리 도메인으로 삼아, 건축 양식 판별 파이프라인을 설계·검증하는 머신비전 프로젝트**

현대오토에버 모빌리티 SW 스쿨 4기 · Team 5조 (이태인, 지태현, 최승재, 허민엽, 최유성)

[📄 최종 발표 자료](ppt/K건축여행도감_최종발표.pdf) · 🎥 데모 영상은 `ppt/K건축여행도감_데모영상.mp4`(용량 문제로 저장소에는 미포함, 로컬 보관)

---

## 한 줄 요약

사진 한 장을 올리면 텀블러의 형태(머그형 / 직선 원통형 / 연속 테이퍼형 / 단차 테이퍼형)를 자동으로 분류하고, 같은 실루엣 규칙이 건축물에도 적용된다는 전제로 "K-건축 여행 도감"이라는 수집형 모바일 데모를 만들었다. **6가지 분류 방법론(룰베이스 → ResNet-18 → EfficientNet-B0 → ConvNeXt-Tiny → Vision Transformer, MobileNetV3-Small 시도)을 같은 데이터·같은 평가 기준으로 공정하게 비교**하는 것이 이 프로젝트의 핵심 — 개별 모델의 정확도보다 **"어떤 방법론이 왜 잘 되고 왜 안 되는지"를 시스템 전체로 검증**하는 데 집중했다.

## 왜 텀블러인가 — 대리 도메인(Proxy Domain) 설계

최종 목표는 **건축 양식 판별**이지만, 라벨링된 건축물 데이터를 직접 수집하기는 현실적으로 어렵다. 대신 **같은 판정 구조를 가진 대리 도메인(텀블러 형태)** 으로 치환해 파이프라인을 먼저 구현·검증했다.

| 항목 | 건축양식 (목표 도메인) | 텀블러 (대리 도메인) |
|---|---|---|
| 판정 유형 | 식별·폐쇄집합 분류 | 동일 |
| 판별 단서 | 정면 파사드의 윤곽 + 부분 요소 | 정면 몸통의 윤곽 + 기울기·단차 |
| 공통 난점 | 클래스 경계 모호, 혼합양식 존재 | 경계 모호, 중간 형태 존재 |
| 촬영 변이 | 각도·거리·조명·가림 | 동일 |
| 데이터 수집 | 직접 수집 어려움 | 수집 가능 |

두 도메인 모두 **"정면이 보여야 판정된다"**는 공통 성질을 가지며, 이 제약이 촬영 프로토콜과 룰베이스 판정 로직에 그대로 반영됐다. 실제로 촬영 각도(원근 왜곡)가 이 프로젝트에서 가장 큰 domain-shift 실패 원인으로 확인됐는데(`docs/troubleshooting.md`), 이는 건축물 파사드 인식에서도 그대로 재현될 문제다.

### 분류 체계 (형태 4종)

| 종류 | 판정 규칙 |
|---|---|
| 머그형 (`mug`) | 높이가 지름의 1.5배 이하 — **손잡이 유무는 보지 않는다** |
| 직선 원통형 (`straight`) | 아래 지름이 위 지름의 95% 이상, 윤곽이 거의 수직 |
| 연속 테이퍼형 (`taper_smooth`) | 아래로 좁아지되 꺾임점 없이 매끄럽게 좁아짐 |
| 단차 테이퍼형 (`taper_step`) | 윤곽에 뚜렷한 꺾임이 1회 이상, 그 지점에서 지름이 줄어듦 |

**우선순위 규칙**: 실루엣이 손잡이보다 우선한다 — 손잡이 달린 테이퍼 텀블러는 머그형이 아니라 연속 테이퍼형. 재질·용량·브랜드는 분류 축에서 제외(형태 하나에만 집중).

---

## 데이터

| 구분 | 수량 | 출처 |
|---|---|---|
| 학습(train) | 515장 | 크롤링(네이버 쇼핑 API 1순위, 부족분은 검색 크롤링) |
| 검증(val, "Test1") | 127장 | 크롤링 — 학습과 같은 도메인의 홀드아웃 |
| 평가(test, "Test2") | 126장 | **직접 촬영** — 학습에 전혀 쓰이지 않은 실사용 사진 |

- **의도된 domain shift 평가**: train/val은 웹 이미지, test는 직접 촬영한 실사진. 정확도가 떨어지는 게 정상이며, **그 하락폭 자체가 비교 지표**다.
- 크롤링 700장 → 라벨 품질 필터링(룰베이스+Mask R-CNN이 둘 다 폴더 라벨과 다르게 판정한 "강한 신호" 제외) → 642장으로 정제. 직접 촬영 158장 → 배경-객체 색상 유사 등 애매한 32장 제외 → 126장.
- 모든 트랙이 **동일한 stratified_split(seed=42)** 를 재사용해서, Test1(Val) 127장이 트랙 간에 완전히 동일하게 재현된다 — 공정 비교의 핵심 장치.

자세한 수집·전처리 과정은 [`docs/dataset-composition.md`](docs/dataset-composition.md), [`docs/data-collection-plan.md`](docs/data-collection-plan.md), [`docs/label-quality-filtering.md`](docs/label-quality-filtering.md) 참고.

---

## 결과 — 6갈래 방법론 비교

| 트랙 | 접근 | Test1(Val) | Test2(실촬영) | 비고 |
|---|---|---|---|---|
| OpenCV 룰베이스 | width profile + convexity defects | 63.0% / F1 0.610 | 18.4% / F1 0.139 | 정면 촬영 가정이 깨지면 구조적으로 취약 — 4트랙 중 하락폭 최대 |
| Mask R-CNN (제로샷) | ResNet50+FPN 세그멘테이션 → 룰베이스 폭 프로파일 재사용 | - | 43.7% / F1 0.466 | 파인튜닝 시도는 오히려 성능 하락(39.7%→36.7%)해서 제로샷 유지 |
| ResNet-18 (+TTA) | 전이학습, 백본 unfreeze + 차등 LR + Test-Time Augmentation | 96.9% | 82.5% / F1 0.829 | 재학습 없이 추론만 바꿔 "공짜 개선" |
| EfficientNet-B0 | timm, compound scaling + MBConv | 96.9% | 88.1% / F1 0.879 | 6갈래 중 Test2 최고 |
| ConvNeXt-Tiny | 2단계 학습(head warmup → LLRD finetune) + RandomPerspective/Erasing | 98.4% | 86.5% / F1 0.867 | val loss 기준 early stopping |
| Vision Transformer (DeiT-Small) | Self-attention, 패치 토큰 기반 | 98.4% | 85.7% / F1 0.856 | 완전 베이스라인(증강·부분freeze 없음)이 오히려 정교한 설계보다 좋았던 역설적 결과 |

> 핵심 발견: 이 프로젝트의 가장 큰 domain shift 원인은 **촬영 각도(원근 왜곡)** — 기하학적 규칙에 의존하는 트랙(룰베이스, Mask R-CNN)일수록 크게 무너지고, **학습된 시각 패턴**(ResNet·EfficientNet·ConvNeXt·ViT)을 쓰는 방식이 훨씬 강건했다. Grad-CAM으로 확인해도 CNN/ViT 계열은 배경이 아니라 물체 본체·손잡이에 정확히 집중하고 있었다(`reports/figures/{resnet18,vit}/gradcam_test2.png`).

전체 실험 과정(시행착오 포함)과 트랙별 상세 수치는 [`docs/experiment-log.md`](docs/experiment-log.md)에 누적 기록돼 있다.

---

## 라이브 데모 2종

### 1. K-Arch Trip (Streamlit) — 사진/실시간 카메라 판정 + 도감 수집

```bash
streamlit run app/demo.py
```

사진 업로드 또는 PC 카메라로 실시간 판정, 판정 근거 히트맵(Grad-CAM), 형태별 해설, 확정된 판정을 모으는 "도감" 기능을 제공한다. 모델은 `checkpoints/model/meta.json` 하나로 교체 가능(코드 수정 불필요).

### 2. K-Trip 모바일 웹 데모 (FastAPI) — 휴대폰에서 바로 체험

```bash
cd k_trip_ios && python -m uvicorn server:app
# 또는 share.bat 실행 → Cloudflare 터널로 공개 링크 + QR 생성
```

아이폰 UI를 흉내 낸 모바일 웹 앱(네이티브 Swift 아님). 발표 당일 QR 코드로 체험자들이 직접 촬영해 도감을 채우는 라이브 데모로 사용했다 — 최종 발표 PDF의 "Live Demo" QR 코드가 바로 이 앱이다.

---

## 아키텍처 — 트랙 병렬 비교 구조

```
src/
├── rule_based/          # OpenCV 룰베이스 — Otsu+Canny→GrabCut 마스크, width profile 판정
├── deep_learning/
│   ├── dl1_maskrcnn/     # Mask R-CNN 세그멘테이션 → rule_based 함수 재사용
│   ├── dl2_resnet/       # ResNet-18/50 전이학습 + TTA + class-weighted loss
│   ├── efficientnet/     # EfficientNet-B0 전이학습
│   ├── convnext/         # ConvNeXt-Tiny, 2단계(warmup→LLRD finetune) 학습
│   ├── vit/              # DeiT-Small(ViT), 부분 fine-tuning + 종횡비 보존 패딩
│   └── custom_cnn/       # 직접 설계한 SimpleCNN (교육용 대조군, 4트랙 비교 제외)
├── data_collection/      # 네이버 쇼핑 API / 검색 크롤링
├── pipeline.py           # meta.json 기반 범용 추론 파이프라인 (torchvision/timm 아키텍처 무관)
├── explanation/          # 형태별 해설 텍스트
└── gradcam*.py           # 판정 근거 히트맵·문장 생성

app/demo.py               # Streamlit 데모
k_trip_ios/                # FastAPI 모바일 웹 데모
```

**트랙 폴더 밖의 코드를 임의로 공유하지 않는다**(유일한 예외: `dl1_maskrcnn`이 `rule_based`의 width-profile 함수를 재사용) — 6갈래를 서로 다른 코드 경로로 유지해야 공정 비교가 성립하기 때문. 모든 트랙이 `--eval-only`, Test1/Test2 평가, `reports/figures/<track>/` 출력 구조를 동일하게 따른다.

---

## 실행 방법

```bash
pip install -r requirements.txt
python src/deep_learning/efficientnet/train.py --eval-only   # 기존 체크포인트로 재평가(재학습 없이)
python src/deep_learning/efficientnet/train.py               # 처음부터 학습
```

각 트랙·데모 앱을 어떻게 재현·검증하는지는 [`docs/testing.md`](docs/testing.md)에 정리했다.

---

## 기술 스택

PyTorch · torchvision · timm (DeiT) · OpenCV (Otsu/Canny/GrabCut) · scikit-learn 스타일 지표 · Streamlit + streamlit-webrtc · FastAPI + WebSocket · matplotlib (학습곡선/confusion matrix/Grad-CAM)

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/experiment-log.md`](docs/experiment-log.md) | 트랙별 실험 결과·시행착오 전체 기록 |
| [`docs/testing.md`](docs/testing.md) | 전체 기능 재현·검증 방법 |
| [`docs/dataset-composition.md`](docs/dataset-composition.md) | 클래스별 데이터 구성 |
| [`docs/data-collection-plan.md`](docs/data-collection-plan.md) | 수집 전략(API/크롤링/직접 촬영 우선순위) |
| [`docs/label-quality-filtering.md`](docs/label-quality-filtering.md) | 라벨 품질 필터링(룰베이스+Mask R-CNN 교차검증) |
| [`docs/evaluation-metrics.md`](docs/evaluation-metrics.md) | 공통 평가 지표 정의 |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | 겪은 문제와 해결 과정 |

## 팀

현대오토에버 모빌리티 SW 스쿨 4기 · Team 5조 — 이태인, 지태현, 최승재, 허민엽, 최유성
