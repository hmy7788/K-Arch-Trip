# K-Trip iOS demo

텀블러 모양의 건축물로 이루어진 대한민국을 여행하며 네 가지 건축 양식을 모으는 도감입니다. 아이폰 외형과 Dynamic Island 내부에 Discover / Camera / My atlas를 구성했습니다. 영어·한국어를 지원합니다.

## 빠른 시작: 시연 서버 띄우기 (Windows)

시연 대상자가 많으면 **여러 PC에서 각각 서버를 띄우고, PC마다 다른 링크(QR)로 사람들을 나눠서 받으면 됩니다.** 각 PC에서 아래 순서를 그대로 따라 하세요.

### 준비물

| 항목 | 설명 |
|---|---|
| Windows 10/11 PC | 인터넷 연결 필요 |
| Python 3.10 이상 | 없으면 [python.org](https://www.python.org/downloads/)에서 3.12를 설치하세요. 설치 첫 화면에서 **Add python.exe to PATH**를 꼭 체크하세요 |
| `best_convnext_v5.pth` | 팀원이 각자 가지고 있는 모델 가중치(약 111MB). Git에는 올리지 않습니다 |

### 1단계: 브랜치 받기

```powershell
git clone -b seungjae/k_trip_ios https://github.com/hmy7788/SmartFactory_Vision_Project1.git
cd SmartFactory_Vision_Project1\k_trip_ios
```

이미 클론했다면 `git fetch origin` → `git switch seungjae/k_trip_ios` → `git pull`만 하면 됩니다.

### 2단계: 가중치 넣기

가지고 있는 `best_convnext_v5.pth`를 아래 위치에 복사하세요.

```
k_trip_ios\checkpoints\best_convnext_v5.pth
```

### 3단계: `setup.bat` 실행 (처음 한 번만)

탐색기에서 `k_trip_ios\setup.bat`을 더블클릭하세요. 아래 작업을 자동으로 합니다.

1. Python 3.10+ 찾기 (3.12 → 3.11 → 3.10 순서로 우선)
2. `k_trip_ios\.venv` 가상환경 생성
3. `requirements.txt` 설치 (PyTorch 때문에 처음에는 **5~10분** 걸릴 수 있습니다)
4. `tools\cloudflared.exe` 다운로드 (Cloudflare 공식 GitHub 릴리스, 버전 고정 + SHA256 검증)
5. 가중치 파일 확인

마지막에 `설치 완료!`가 보이면 성공입니다. 여러 번 실행해도 안전합니다. 환경이 꼬였다면 PowerShell에서 `.\setup.bat -Recreate`로 `.venv`를 새로 만들 수 있습니다.

### 4단계: `share.bat` 실행 (시연할 때마다)

`k_trip_ios\share.bat`을 더블클릭하세요.

1. 서버를 켜고 모델 로드가 끝날 때까지 기다립니다 (`[1/3]`)
2. Cloudflare 임시 터널로 `https://xxxx.trycloudflare.com` 공개 링크를 만듭니다 (`[2/3]`)
3. 외부에서 링크가 열리는지 확인한 뒤 **링크와 QR 코드**를 보여 줍니다 (`[3/3]`)

QR은 콘솔에도 출력되고, PC 이름이 적힌 큰 이미지(`share_qr.png`)로도 자동으로 열립니다. 그 이미지를 모니터에 띄워 두거나 인쇄해서 체험자가 폰으로 찍게 하면 됩니다. 링크 텍스트는 `share_link.txt`에도 저장됩니다.

> **이 창을 닫으면 링크가 사라집니다.** 시연하는 동안 계속 켜 두세요. 종료는 창 닫기 또는 `Ctrl+C`입니다.
> 링크는 **실행할 때마다 새로 바뀝니다.** QR은 그날 `share.bat`을 켠 뒤에 새로 띄우세요.

체험자는 설치할 것이 없습니다. 폰이나 노트북으로 링크를 열고 Camera → Enable camera에서 카메라 권한을 허용하면 됩니다. https 링크라서 아이폰에서도 카메라가 동작합니다.

### 여러 PC로 나눠 운영할 때

- PC마다 `share.bat`을 켜면 PC마다 **다른 링크**가 나옵니다. QR 이미지 아래에 PC 이름(기본값은 컴퓨터 이름)이 적혀 있어 구분할 수 있습니다. 이름을 바꾸려면 PowerShell에서 `.\share.bat -Name 부스A`처럼 실행하세요.
- 추론은 PC 한 대에서 한 번에 하나씩 처리합니다. CPU 기준 초당 약 4회라서, **한 PC에 동시 접속자가 늘면 모두 느려집니다.** 느려지면 PC를 추가하고 새 QR로 사람들을 나눠 주세요.
- 도감(My atlas)은 체험자 브라우저에 **링크별로** 저장됩니다. 다른 PC 링크로 옮기면 도감이 이어지지 않으니 한 사람은 한 링크만 쓰게 안내하세요.
- 8765 포트가 이미 사용 중이면(예: `run.bat`이 켜져 있음) 자동으로 8766, 8767 …을 사용합니다. 한 PC에서 `share.bat`을 두 번 켤 수도 있지만 CPU를 나눠 쓰므로 빨라지지는 않습니다.

### 문제 해결

| 증상 | 해결 |
|---|---|
| `Python 3.10 이상을 찾지 못했습니다` | Python 3.12 설치 시 **Add python.exe to PATH** 체크 후 setup.bat 재실행 |
| `패키지 설치 실패` | 인터넷·회사 프록시 확인 후 setup.bat 재실행 (이미 받은 패키지는 건너뜀) |
| `가상환경(.venv)이 없습니다` / `cloudflared.exe가 없습니다` | setup.bat을 먼저 실행 |
| `모델 가중치가 없습니다` | 2단계 위치에 `best_convnext_v5.pth` 복사 |
| `서버가 준비되지 않았습니다` | 화면에 나온 로그 확인. 전체 로그는 `k_trip_ios\logs\server.err.log` |
| `60초 안에 링크를 받지 못했습니다` | 행사장/회사 네트워크가 Cloudflare 터널을 막는 경우입니다. 폰 핫스팟으로 바꿔 보세요. 로그: `logs\tunnel.err.log` |
| `아직 외부에서 확인되지 않았습니다` | 새 링크가 퍼지는 데 시간이 걸린 경우입니다. 30초쯤 뒤 폰으로 열어 보세요 |
| 폰에서 카메라가 안 켜짐 | 브라우저의 카메라 권한 허용 확인. 카카오톡 등 앱 내장 브라우저라면 Safari/Chrome으로 열기 |

### 이 PC에서만 쓸 때 (링크 없이)

`run.bat`을 실행하고 [http://localhost:8765](http://localhost:8765)를 여세요. `.venv`가 있으면 자동으로 사용합니다. 카메라는 localhost 또는 HTTPS에서만 동작하므로, 같은 와이파이의 폰에서 `http://PC-IP:8765`로 접속하는 방식은 카메라가 동작하지 않습니다. 다른 기기에서는 `share.bat` 링크를 쓰세요.

macOS/Linux에서는 `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` 후 `python -m uvicorn server:app --host 127.0.0.1 --port 8765 --ws-max-size 6291456`으로 실행하고, 공개 링크가 필요하면 [cloudflared](https://github.com/cloudflare/cloudflared/releases)를 설치해 `cloudflared tunnel --url http://127.0.0.1:8765`를 따로 실행하세요.

### 파일 구성 (실행 관련)

| 파일 | 역할 |
|---|---|
| `setup.bat` → `tools/setup.ps1` | 첫 설치 (.venv, 패키지, cloudflared, 가중치 확인) |
| `share.bat` → `tools/share.ps1` | 서버 + 공개 링크 + QR |
| `tools/qr.py` | QR 콘솔 출력과 `share_qr.png` 생성 |
| `run.bat` → `run.ps1` | 로컬 전용 실행 |
| `logs/`, `tools/cloudflared.exe`, `share_qr.png`, `share_link.txt`, `.venv/` | 실행 중 생기는 파일. Git에 올리지 않음 |

## 기능

- 브라우저 카메라 30fps 요청, 실제 재생 FPS와 추론 갱신 속도 분리 표시
- WebSocket으로 한 프레임씩 전송하여 대기열 누적 방지
- 별도 촬영 버튼과 사진 업로드, 클래스별 확률, Grad-CAM
- 기존 팀 프로젝트의 양식 명칭·주요 특징·구조적 이유·건축 사조·대표 건축물
- 도감 양식 아이콘 선택, 발견 여부와 수집 수, 사진 필터와 상세 설명 연동
- IndexedDB에 사진과 분석 저장, 새로고침 유지, 삭제 및 분석 다시 보기

## 모델과 추론

- torchvision ConvNeXt-Tiny, `best_convnext_v5.pth` (약 111.4 MB)
- 클래스 순서: `mug`, `straight`, `taper_smooth`, `taper_step`
- 전체 이미지 Resize((224,224), bilinear) → ImageNet Normalize
- GPU 지원 PyTorch와 GPU가 있으면 CUDA, 그 외에는 CPU 자동 선택
- 오른쪽 모델명은 `/api/health`의 실제 모델 정보에서 표시
- 새 캡처에는 모델 아키텍처와 리사이즈 방식도 저장. 기존 도감 사진은 자동 재분류하지 않음
- 다른 체크포인트 폴더는 `KTRIP_CHECKPOINT` 환경변수로 지정

카메라 프레임은 최대 640px JPEG로 전송하고 추론 응답 80ms 후 다음 프레임을 보냅니다. 개발 PC의 CPU에서 추론 약 142ms, 확률 갱신 약 4Hz 전후가 예상되며 기기와 부하에 따라 달라집니다. 카메라 재생과 추론 갱신 속도는 별도입니다. 캡처는 최대 1280px 사진에 Grad-CAM을 계산합니다. 카메라 정지·탭 이동·백그라운드 전환 시 카메라 트랙을 종료합니다.

70% 미만 결과는 잠정 분류로 저장하고 4종 수집 달성에서 제외합니다. 이 모델은 사물 부재나 미학습 사물을 검출하지 않습니다. Grad-CAM은 모델의 주목 영역이며 정답의 증거가 아닙니다. 형태 해설은 세계관의 양식 안내이며 실제 윤곽을 측정한 결과와 구분합니다.

## 코드 구성과 출처

- `server.py`: FastAPI 캡처 API 및 WebSocket 추론. 모델 접근을 직렬화하여 Grad-CAM과 실시간 추론 충돌 방지
- `static/`: 빌드 과정 없는 HTML/CSS/JavaScript UI. `profiles.js`는 기존 `src/explanation/class_profiles.yaml`의 한국어 내용을 옮기고 영어 번역을 추가
- `src/pipeline.py`: 팀 웹 데모의 메타데이터 기반 로더를 재사용. 히트맵은 실제 모델 입력 이미지에서 복원
- `src/gradcam.py`: 팀 Grad-CAM 수식을 사용하고 임시 hook / autograd.grad로 추론 후 훅이 남지 않도록 수정
- `checkpoints/meta.json`: 팀 ConvNeXt 추론 코드의 모델 구성·전처리·클래스 순서

사진은 서버에 영구 저장하지 않으며 같은 브라우저·origin의 IndexedDB에 보관됩니다. 브라우저 데이터 삭제 시 도감도 사라집니다.

## 검증

모델 파일 배치와 의존성 설치 후 `k_trip_ios`에서 실행하세요.

```powershell
.\.venv\Scripts\python.exe -m unittest test_demo -v
```

실제 모델로 API, 확률 합, Grad-CAM 이후 반복 추론, 잘못된/과대 입력 거부, 학습 전처리 일치를 검증합니다. 브라우저에서는 사진 분석·히트맵·도감 영구 저장·양식 필터·상세 설명·한영 전환을 확인했습니다. 개발 중 내장 브라우저의 카메라 권한이 거부되어 실제 카메라 FPS는 측정하지 못했습니다.
