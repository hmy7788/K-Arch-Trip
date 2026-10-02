# 텀블러 형태 분류 및 설명 시스템

Computer Vision Project · 2026

사용자가 텀블러 이미지를 업로드하면 형태를 자동으로 분류하고, 해당 종류에 대한 설명을 제공하는 이미지 인식 시스템. **향후 건축 양식 판별로 확장 가능한 구조로 설계.**

---

## 프로젝트 개요

### 왜 텀블러인가 — 대리 도메인(proxy domain) 설계

이 프로젝트의 최종 목표는 **건축 양식 판별**이지만, 실제 건축물 데이터를 라벨링해서 직접 수집하는 것은 현실적으로 어려움. 대신 **동형 구조를 가진 대리 도메인(텀블러 형태)** 으로 치환해 파이프라인을 먼저 구현·검증한다.

> **검증 대상은 모델이 아니라 시스템이다.**

| 항목 | 건축양식 (목표 도메인) | 텀블러 (대리 도메인) |
|---|---|---|
| 판정 유형 | 식별·폐쇄집합 분류 | 동일 |
| 판별 단서 | 정면 파사드의 윤곽 + 부분 요소(창·지붕·기둥) | 정면 몸통의 윤곽 + 기울기·단차 |
| 공통 난점 | 클래스 경계 모호, 혼합양식 존재 | 경계 모호, 중간 형태 존재 |
| 촬영 변이 | 각도·거리·조명·가림 | 동일 |
| 데이터 수집 | 직접 수집 어려움 | 수집 가능 |

두 도메인 모두 **"정면이 보여야 판정된다"**는 공통 성질을 가지며, 이 제약이 촬영 프로토콜에 그대로 반영됨. (광각 왜곡 보정, 배경 분리, 위치 정보 등은 대리 도메인에서 검증되지 않는 항목으로, 목표 도메인 적용 시 추가 필요.)

### 분류 체계

**형태 (4종)** — 몸통 실루엣이라는 하나의 축으로 전부 갈림. 판정 규칙은 문장·숫자로 고정.

| 종류 | 판정 규칙 | 헷갈릴 상대 |
|---|---|---|
| 머그형 | 높이가 지름의 1.5배 이하 (손잡이 유무는 보지 않는다) | — |
| 직선 원통형 | 아래 지름이 위 지름의 95% 이상, 윤곽이 거의 수직 | 연속 테이퍼형 |
| 연속 테이퍼형 | 아래로 좁아지되 꺾임점 없이 매끄럽게 좁아짐 | 직선 원통형, 단차 테이퍼형 |
| 단차 테이퍼형 | 윤곽에 뚜렷한 꺾임이 1회 이상 있고 그 지점에서 지름이 줄어듦 | 연속 테이퍼형 |

**우선순위 규칙**: 실루엣이 손잡이보다 우선한다. 손잡이 달린 테이퍼 텀블러는 머그형이 아니라 연속 테이퍼형.
**애매하면 라벨을 붙이지 않고 `_hold` 폴더로 분류** — 주 1회 팀원 전원이 모여 함께 판정.
**클래스로 쓰지 않는 축**: 손잡이 유무, 재질, 용량, 용도, 브랜드 (카메라로 볼 수 없거나 형태와 무관)

> **재질 분류는 이번 스코프에서 제외.** 표면 반사·질감 신호가 불안정해 규칙화·학습 난이도가 높고, 형태 축 하나만으로도 대리 도메인 검증 목적을 충분히 달성할 수 있다고 판단해 형태 분류에 집중.

### 핵심 기능

1. **텀블러 형태 판별** — 이미지 업로드 시 4가지 형태를 자동 분류하고, 종류별 설명(형태 특징, 용도, 관련 정보) 제공
2. **데이터 수집 및 크롤링** — 직접 촬영(Test 전량) + 네이버 쇼핑 API(1순위)·검색 크롤링(부족분)으로 클래스당 200~300장 확보, 회전·반전·색상 변환으로 증강
3. **(선택) 3D 공간 분류 시각화** — 분류된 텀블러 이미지를 3D 공간에 같은 형태끼리 배치해 탐색하는 웹 뷰어 (Three.js + CLIP 임베딩)

### 시스템 파이프라인

```
이미지 업로드 → 전처리(Resize, Normalize) → 모델 추론 → 결과 출력(Confidence) → 설명 출력(형태 설명)
```

### 데이터 구성

| 구분 | 장수 | 출처 | 용도 |
|---|---|---|---|
| Train | 320 (클래스당 80) | 크롤링 | 학습 |
| Validation | 80 (클래스당 20) | 크롤링 | 하이퍼파라미터 조정 |
| Test | 100 (클래스당 25) | 직접 촬영 | 최종 평가 (1회만) |

- **개체 단위 분할**: 같은 물건의 사진이 train/test에 나뉘어 들어가면 안 됨 (형태가 아니라 물건을 외우게 됨). 물건 A는 통째로 train, 물건 B는 통째로 test.
- **Test는 장수보다 개체 수**: 클래스당 서로 다른 물건 10개 이상 × 2~3장.
- **발표 포인트**: train=웹 이미지, test=실사용 이미지 → 단순 분할이 아니라 **도메인 이동(domain shift) 평가**. 정확도가 낮게 나오는 것이 정상이며 의도된 설계.

수집 출처 우선순위: ① 네이버 쇼핑 API (공식, 상품 카테고리가 라벨 역할) → ② 검색 크롤링 (부족분만) → ③ 직접 촬영 (Test 전량). 개인거래 플랫폼(당근 등)은 약관·사생활 문제로 수집하지 않음.

### 촬영 프로토콜 (Test)

| 변수 | 조건 |
|---|---|
| 각도 | 정면 중심(절반 이상) / 45도 / 위에서 살짝 |
| 배경 | 책상 / 바닥 / 손에 든 것 — 최소 3종 |
| 조명 | 실내등 / 창가 자연광 / 어두운 곳 |
| 상태 | 뚜껑 열림 / 닫힘 |

배경이 한 종류면 모델이 배경을 학습해버림 (흰 책상에서만 찍으면 "흰 책상 위 물체=텀블러"로 학습). 정면 비율을 높이는 이유는 단차·테이퍼 기울기가 각도에 따라 왜곡/소실되기 때문 — 건축양식의 정면 파사드 요구사항과 같은 성질이라 이 제약 자체가 치환 근거가 됨.

### 머신비전 구현 방법 — 4갈래 병렬 비교 (형태 분류)

| 트랙 | 접근 | 형태 분류 방법 |
|---|---|---|
| OpenCV 룰베이스 | 고전 영상처리 | width profile(폭 프로파일) + `cv2.convexityDefects`(손잡이 검출) |
| Mask R-CNN / Faster R-CNN (DL1) | Instance Segmentation | ResNet50+FPN → Mask 이진화 후 Rule-base와 유사한 기하 특징 추출 |
| ResNet-18, 50 (DL2) | 전이학습 | pretrained + 백본 레이어만 fine-tuning, 18→50 순차 비교 |
| EfficientNet-B0 | 전이학습 | timm 기반 EfficientNet-B0 fine-tuning (분류) |
| 3D 모델 (선택) | MV-DUSt3R+ (Meta) | 촬영 사진(4장 내외)으로부터 3D 포인트클라우드/메쉬 생성 |

네 갈래 모두 같은 형태 분류 문제를 서로 다른 방법론(룰베이스 / 세그멘테이션 기반 / 순수 분류 전이학습 / 효율적 아키텍처)으로 풀어 성능·구현 난이도를 비교하는 것이 핵심 — "검증 대상은 시스템"이라는 원칙을 각 트랙이 공정하게 비교 가능한 구조로 재확인.

### 일정 (5일)

| Day | 내용 |
|---|---|
| 1 | 환경 설정 + 데이터 수집 계획 — 클래스 확정, 크롤링 키워드 선정, 직접 촬영 시작 |
| 2 | 데이터 수집 완료 + 증강 — 웹 크롤링, 클래스당 200장↑ 확보, 회전·플립·색상 증강 |
| 3 | 모델 학습 — 룰베이스 완성 + Mask R-CNN(dl1)·ResNet-18/50(dl2)·EfficientNet-B0(4트랙) fine-tuning, 검증 정확도 비교 |
| 4 | 설명글 + 파이프라인 통합 + UI — 형태별 설명 작성, Gradio UI 구현 |
| 5 (선택) | 3D 시각화 + 데모 완성 — Three.js 임베딩 뷰어, 엣지 케이스 처리, 발표 시나리오 준비 |

---

## 디렉토리 구조 추천

```
tumbler-classification/
├── data/
│   ├── raw/                       # 크롤링·촬영 원본 (미가공)
│   ├── train/
│   │   ├── straight/              # 직선 원통형
│   │   ├── taper_smooth/          # 연속 테이퍼형
│   │   ├── taper_step/            # 단차 테이퍼형
│   │   └── mug/                   # 머그형
│   ├── val/
│   │   ├── straight/  taper_smooth/  taper_step/  mug/
│   ├── test/
│   │   ├── straight/  taper_smooth/  taper_step/  mug/
│   └── _hold/                     # 애매해서 보류된 이미지
│
├── src/
│   ├── data_collection/
│   │   ├── naver_shopping_api.py  # 1순위 수집
│   │   ├── search_crawler.py      # 2순위 (부족분만)
│   │   └── augmentation.py        # 회전·플립·색상 증강
│   │
│   ├── rule_based/                # OpenCV 룰베이스 트랙
│   │   └── shape_classifier.py    # width profile + convexity defects
│   │
│   ├── models/
│   │   ├── dl1_maskrcnn/          # Mask/Faster R-CNN (형태 분류)
│   │   ├── dl2_resnet/            # ResNet-18 → 50
│   │   └── efficientnet/
│   │
│   ├── reconstruction_3d/         # (선택) MV-DUSt3R+ 기반 3D 생성
│   │
│   ├── explanation/               # 형태별 설명 텍스트 데이터
│   │
│   └── pipeline.py                # 통합 추론 파이프라인
│
├── app/
│   └── gradio_app.py              # 업로드 → 분류 → 설명 UI
│
├── notebooks/
│   ├── 01_data_diversity_check.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_comparison.ipynb  # Rule-base vs DL1 vs DL2 vs EfficientNet 비교
│
├── checkpoints/                   # 학습된 모델 가중치
├── reports/
│   └── figures/                   # confusion matrix, 비교 그래프 등
│
├── requirements.txt
└── README.md
```

**폴더명은 경로 문제를 피하기 위해 영문**으로 둔다 (`straight` / `taper_smooth` / `taper_step` / `mug`). `src/` 아래를 방법론별(`rule_based`, `dl1_maskrcnn`, `dl2_resnet`, `efficientnet`)로 명확히 나눈 이유는, 4갈래 접근을 병렬로 비교하는 프로젝트 구조상 각 트랙의 코드가 서로 섞이지 않아야 `notebooks/03_model_comparison.ipynb`에서 공정하게 비교할 수 있기 때문.

---

## Claude Code 프롬프트

```
「텀블러 형태 분류 및 설명 시스템」 프로젝트를 시작하려고 해.
아래 설계를 기반으로 프로젝트 스캐폴딩(디렉토리, 코드 스켈레톤, 환경설정)을 먼저 구축해줘.
실제 데이터(크롤링·촬영)는 아직 없으니, 데이터 없이도 진행 가능한 부분부터 진행해줘.

## 프로젝트 배경
최종 목표는 "건축 양식 판별"이지만, 건축물 데이터 직접 수집이 어려워 동형 구조의 대리 도메인인
"텀블러 형태 판별"로 치환해 파이프라인을 먼저 구현·검증한다. 검증 대상은 모델이 아니라 시스템이다.
재질 분류는 이번 스코프에서 제외하고 형태 분류(4종)에만 집중한다.

## 분류 체계
- 형태(4종): 직선 원통형(straight) / 연속 테이퍼형(taper_smooth) / 단차 테이퍼형(taper_step) / 머그형(mug)
- 판정 규칙:
  1) 높이가 지름의 1.5배 이하면 머그형 (손잡이 유무는 무시)
  2) 아래 지름이 위 지름의 95% 이상, 윤곽 거의 수직이면 직선 원통형
  3) 꺾임점 없이 매끄럽게 좁아지면 연속 테이퍼형
  4) 윤곽에 뚜렷한 꺾임이 1회 이상 있으면 단차 테이퍼형
  5) 우선순위: 실루엣이 손잡이보다 우선 (손잡이 달린 테이퍼는 머그형이 아니라 연속 테이퍼형)

## 1. 디렉토리 구조 생성
아래 구조로 폴더를 생성해줘 (README.md의 "디렉토리 구조 추천" 섹션과 동일):
tumbler-classification/
├── data/{raw, train/{straight,taper_smooth,taper_step,mug}, val/{...}, test/{...}, _hold}/
├── src/{data_collection, rule_based, models/{dl1_maskrcnn, dl2_resnet, efficientnet},
│        reconstruction_3d, explanation}/
├── app/
├── notebooks/
├── checkpoints/
└── reports/figures/

## 2. requirements.txt 작성
Python 3.10 기준: opencv-python, torch, torchvision, timm(EfficientNet/ViT용),
scikit-learn(HOG/SVM baseline), pandas, numpy, matplotlib, seaborn, gradio,
naver-shopping API 호출용 requests, Pillow

## 3. src/rule_based/shape_classifier.py 구현
- 텀블러 실루엣 마스크를 입력받아, 세로 방향 N등분(기본 10구간) 폭 프로파일을 계산하는 함수
- cv2.convexHull + cv2.convexityDefects로 손잡이(비대칭 돌출부) 검출 함수
- 위 판정 규칙(1~5)을 그대로 구현한 classify_shape() 함수
- 머그형 판별을 "손잡이 유무"가 아니라 "높이/지름 비율 1.5 이하"로 우선 판정하도록 주의해서 구현

## 4. src/data_collection/ 스켈레톤
- naver_shopping_api.py: 네이버 쇼핑 API로 클래스별 키워드 검색 후 이미지 다운로드하는 함수 (API 키는 환경변수로)
- augmentation.py: 회전·플립·색상 변환 증강 함수 (torchvision.transforms 기반)

## 5. src/models/ 스켈레톤
- dl2_resnet/train.py: torchvision의 사전학습 ResNet-18/50을 불러와 마지막 FC layer만
  우선 fine-tuning하는 형태 분류 학습 스크립트 (백본 freeze 옵션 포함)
- efficientnet/train.py: timm 라이브러리로 EfficientNet-B0 fine-tuning 스크립트 (형태 분류)
- dl1_maskrcnn/: torchvision의 Mask R-CNN(ResNet50+FPN)을 로드하고, 예측된 Mask를
  이진화해서 src/rule_based/shape_classifier.py의 width profile 함수를 재사용하도록 연결하는 구조
  (형태만 판별, 재질 헤드는 만들지 않음)

## 6. src/pipeline.py 구현
이미지 업로드 → 전처리(Resize, Normalize) → 모델 추론 → Confidence 포함 형태 분류 결과 반환 →
형태별 설명 텍스트(src/explanation/에서 로드) 반환하는 통합 함수 인터페이스 정의
(내부 모델은 아직 학습 전이므로 더미 응답으로 동작 검증)

## 7. app/gradio_app.py
이미지 업로드 → pipeline.py 호출 → 형태/Confidence/설명을 보여주는 간단한 Gradio UI

## 8. README.md
이미 작성된 프로젝트 개요·분류 체계·데이터 구성·촬영 프로토콜·일정을 그대로 유지하고,
실행 방법(설치, 데이터 배치, 학습, 데모 실행 순서) 섹션만 추가해줘.

데이터 의존적인 부분은 더미/샘플 이미지로 동작 검증까지 해줘.
```
