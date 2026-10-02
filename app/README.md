# K-Arch Trip 대시보드 — 어디를 고치면 되나

실행: 저장소 루트에서 `app\run_demo.bat` (윈도우) 또는 `streamlit run app/demo.py`

## 고치고 싶은 것 → 갈 파일

| 고치고 싶은 것 | 파일 | 찾을 이름 |
|---|---|---|
| **확신도 막대그래프** (색·길이·순서·기준선) | `app/demo.py` | `probs_chart()` · 색은 바로 위 `BAR_ACCENT` / `BAR_MUTED` |
| 판정 카드 (🟢 제목·확신도 문구·처리 시간) | `app/demo.py` | `render_verdict()` |
| 근거 히트맵이 **화면에 놓이는 방식** (문구·위치) | `app/demo.py` | `render_gradcam()` |
| 근거 히트맵 **계산**(히트맵 색·투명도·어느 층을 볼지) | `src/gradcam.py` | `overlay_cam()` · `target_layer()` |
| **판정 근거 문장** (📍 어디를 봤나, 확신도 설명) | `src/gradcam_reason.py` | `_reason_<클래스>()` · `_confidence_tier()` |
| 양식 해설 **문구** (구조적 이유·사조·대표 건축물) | `src/explanation/class_profiles.yaml` | 클래스 이름으로 찾기 |
| 결과 카드의 **주요 특징 칩** | `src/explanation/class_profiles.yaml` | 각 클래스의 `features:` |
| 해설 **항목 이름**("대표 건축물" 같은 제목) | `src/explanation/profiles.py` | `FIELD_LABELS` |
| 양식 **이름**(우리말/영문) | `src/explanation/class_profiles.yaml` | `ko:` / `title:` |
| 판정/보류를 가르는 **규칙** | `src/explanation/profiles.py` | `decide()` |
| 앱 이름·태그라인·소개 문장 | `app/demo.py` | `APP_NAME` · `TAGLINE` · `SUBTITLE` |
| 화면 껍데기 (헤더 바·결과 카드·칩 모양) | `app/demo.py` | `st.markdown("""<style>` 블록의 `.ka-*` |
| **도감** (목표 칸 수·썸네일 크기·담는 규칙) | `app/demo.py` | `DEX_GOAL` · `THUMB_W` · `dex_add()` · `render_dex_button()` |
| 도감 화면 · 양식 칩 **필터** | `app/demo.py` | `with tab_dex:` · `st.pills(..., key="dexfilter")` |
| 사이드바 (기준 슬라이더·판정 간격·히트맵 켜기) | `app/demo.py` | `# ---- 사이드바` 블록 |
| 카메라 해상도 | `src/sources.py` | `CAPTURE_SIZE` |
| 탭 이름, 탭 구성 | `app/demo.py` | `st.tabs([...])` |
| 사진 업로드 화면 | `app/demo.py` | `with tab_photo:` |
| 실시간 화면 (카메라 고르기·버튼) | `app/demo.py` | `with tab_live:` |
| 실시간 **루프** (몇 초마다 판정, 화면 몇 fps) | `app/demo.py` | `# ---- 실행: 이 컴퓨터 카메라` |
| 카메라 열기·프레임 읽기 | `src/sources.py` | `webcam_frames()` · `first_camera()` |
| (안 씀) 브라우저 카메라 webrtc | `src/live.py` | 지금은 화면에서 안 쓴다 — 되살릴 때만 |
| **모델 바꾸기** (구조·전처리·클래스 순서·가중치 파일명) | `checkpoints/model/meta.json` | — 코드는 안 건드린다 |
| 모델 로드·전처리·추론 | `src/pipeline.py` | `Pipeline` · `preprocess()` |
| 정확도 검사 도구 | `tools/check_model.py` | `PREFERRED_FOLDER`(검사할 사진 폴더) |

## 큰 그림

```
app/demo.py          화면 전부. 여기만 Streamlit을 안다.
  └ src/pipeline.py    사진 → 판정 (meta.json + 가중치)
          └ src/gradcam.py        판정 근거 히트맵
      └ src/gradcam_reason.py 히트맵 → 근거 문장
  └ src/explanation/   판정 → 사람이 읽을 해설
  └ src/sources.py     이 컴퓨터 카메라
  └ src/live.py        브라우저 카메라 (webrtc, 선택)
```

`src/` 안의 파일들은 Streamlit을 import하지 않는다 — 브라우저 없이 테스트할 수 있게.
화면에 관한 것은 전부 `app/demo.py` 하나에 있다.

## 모델 넣는 법 (checkpoints/model/)

1. `meta.json` — 지금은 팀 ConvNeXt-Tiny 기준. 모델을 바꾸면 `meta.example.json`을 보고
   library/arch · img_size/resize/mean/std · classes 순서 · weights 파일명을 고친다.
2. 가중치(`best_convnext_v5.pth`, 111MB)를 같은 폴더에 넣는다. git에는 안 올라간다 — 직접 받는다.
3. `check_model.bat` — 라벨된 사진 폴더로 정확도를 재고, `--compare`로 팀원 `inference.py`와
   같은 답인지 대조한다. 라벨 일치 100%·확률 차이 ≈0 이면 제대로 붙은 것.
   값이 틀리면(가중치 키·클래스) 로드할 때 에러가 난다. **전처리(resize)만은 틀려도 에러가 안 나고
   점수만 떨어지므로** 이 검사가 필요하다.
4. `app\run_demo.bat`.

## 실시간 탭 — 카메라

이 컴퓨터에 달린 카메라 하나만 쓴다. cv2로 직접 열고(`src/sources.py`), 열리는 카메라를 알아서
찾는다 — 고르는 칸은 없다. 네트워크를 안 타므로 붙기만 하면 끊길 일이 없다.

영상과 판정은 분리돼 있다. 화면은 들어오는 대로 갱신하고, 판정은 사이드바의 간격대로만 돈다.
프레임마다 판정하면 한 장에 0.3초씩 걸려 영상이 끊긴다.

**캡처**를 누르면 영상이 멈추고 그 한 장에 대해 판정·근거 히트맵·해설이 아래에 펼쳐진다.
영상이 도는 동안에는 버튼을 누를 수 없기 때문에(파이썬이 프레임 루프에 붙잡혀 있다) 어차피 멈춘다.

### 브라우저 카메라(webrtc)는 왜 뺐나

팀원이 자기 기기 카메라로 찍게 하려고 붙였다가 뺐다. 붙이는 환경마다 다르게 깨졌다:
연결이 몇 초 만에 끊기고(CPU 경합), STUN 조회가 방화벽에 막혀 매달리고, 터널 뒤에서는
컴포넌트 JS를 `localhost:8501`에서 찾다 실패했다. 시연에 걸 만한 안정성이 아니었다.

코드는 `src/live.py`와 `share_https.bat`·`get_cloudflared.bat`에 남아 있다. 다시 시도하려면
`app/demo.py`의 실시간 탭에 분기를 되살리면 된다. 그때 알아야 할 것:
Streamlit은 페이지에 서버 주소를 박아넣으므로 터널 뒤에서는
`--browser.serverAddress=<공개 호스트> --browser.serverPort=443`을 주고 띄워야 한다.
