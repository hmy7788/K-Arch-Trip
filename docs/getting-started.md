# 실행 방법

저장소 루트에서 실행하는 걸 기준으로 적는다. 재현·검증 절차(기대 수치, 알려진 함정)는 [docs/testing.md](testing.md)를 따로 참고.

## 1. 설치

```bash
pip install -r requirements.txt
```

Python 3.10 기준. GPU가 있으면 `torch`가 자동으로 CUDA를 쓴다. GPU가 없는 환경이면 `requirements.txt`의 `--extra-index-url` 줄을 지우고 `pip install torch torchvision`으로 CPU 빌드를 받으면 된다.

## 2. 데이터 배치

이미 정제된 데이터가 `data/preprocess/<class>/`(학습·검증용, 642장)와 `data/test1_orientation_backup/<class>/`(직접 촬영 평가용, 126장)에 있어야 한다. 클래스 폴더명은 `straight` / `taper_smooth` / `taper_step` / `mug` 4개로 고정.

원본부터 다시 만들려면:

```bash
# 1순위: 네이버 쇼핑 API, 2순위: 검색 크롤링 (docs/data-collection-plan.md)
python src/data_collection/search_crawler.py --class all

# 라벨 품질 필터링(룰베이스+Mask R-CNN 교차검증) 후 data/preprocess/ 생성
# — 절차는 docs/label-quality-filtering.md 참고
```

## 3. 모델 학습

트랙마다 스크립트 위치만 다르고 사용법은 동일하다(`--eval-only`로 재학습 없이 기존 체크포인트 재평가도 가능).

```bash
# ResNet-18 (백본 unfreeze 권장, TTA는 평가에만 적용)
python src/deep_learning/dl2_resnet/train.py --model resnet18 --no-freeze-backbone --epochs 20

# EfficientNet-B0
python src/deep_learning/efficientnet/train.py --no-freeze-backbone --epochs 15

# ConvNeXt-Tiny
python src/deep_learning/convnext/train.py

# Vision Transformer (DeiT-Small) — 완전 베이스라인
python src/deep_learning/vit/train.py --baseline

# 룰베이스는 학습이 필요 없다(규칙 기반) — 바로 평가로 넘어가면 됨
```

체크포인트는 `checkpoints/<모델명>_shape.pth`, 학습 곡선·confusion matrix는 `reports/figures/<트랙>/`에 저장된다.

## 4. 평가 / 추론

```bash
# 기존 체크포인트로 Test1(Val)·Test2(실촬영) 재평가, 재학습 없음
python src/deep_learning/efficientnet/train.py --eval-only

# 판정 근거 히트맵(Grad-CAM)
python src/deep_learning/dl2_resnet/gradcam.py --split test2 --per-class 4 --model resnet18
python src/deep_learning/vit/gradcam.py --split test2 --per-class 4
```

## 5. 데모 앱 실행

### K-Arch Trip (Streamlit)

```bash
streamlit run app/demo.py
```

`checkpoints/model/meta.json`이 가리키는 모델(기본: EfficientNet-B0)을 불러와 사진 업로드·실시간 카메라 판정을 제공한다. 다른 모델로 바꾸려면 코드 수정 없이 `meta.json`의 `arch`/`weights`만 바꾸면 된다.

### K-Trip 모바일 웹 데모 (FastAPI)

```bash
cd k_trip_ios
python -m uvicorn server:app          # 로컬에서만 접속
# 또는
share.bat                             # Cloudflare 터널로 공개 링크 + QR 생성 (Windows)
```

`k_trip_ios/checkpoints/meta.json`이 공식 ConvNeXt-Tiny 체크포인트를 가리키도록 돼 있다. 휴대폰 카메라로 바로 체험하려면 `share.bat`으로 공개 링크를 만들어야 한다(로컬호스트는 외부 기기에서 접속 불가).

## 자주 겪는 문제

- **체크포인트가 없다는 에러** — `--eval-only`나 데모 앱은 `checkpoints/`에 해당 `.pth` 파일이 있어야 동작한다. 이 파일들은 용량 때문에 git에 올리지 않으므로(`.gitignore`), 먼저 3번(모델 학습)을 돌려서 직접 만들거나 팀원에게 전달받아야 한다.
- **한글 경로 관련 `cv2.imread`/`cv2.imwrite` 실패** — 저장소 경로 자체가 한글이면 OpenCV가 조용히 실패한다. 직접 스크립트를 수정할 때는 `np.fromfile`+`cv2.imdecode` / `cv2.imencode`+`tofile` 패턴을 쓸 것(`docs/troubleshooting.md` 참고).
- 그 외 실행 중 오류는 [docs/troubleshooting.md](troubleshooting.md)에 과거 사례가 정리돼 있다.
