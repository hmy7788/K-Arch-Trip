"""브라우저 카메라로 실시간 판정 — streamlit-webrtc가 깔려 있을 때만 쓴다.

무엇이 다른가: 지금까지의 웹캠은 **서버(streamlit을 켠 컴퓨터)**의 카메라를 열었다. 그래서 팀원이
내 주소로 접속하면 팀원 화면에 내 카메라가 보였다. webrtc는 **보는 사람의 브라우저** 카메라를 써서
영상을 서버로 보낸다 — 각자 자기 카메라로 판정받는다.

카메라 권한의 전제: 브라우저는 주소가 https이거나 localhost일 때만 카메라를 내준다(보안 컨텍스트).
같은 와이파이의 http://192.168.x.x:8501로 들어오면 권한 창이 아예 안 뜬다. 코드로 우회할 수 없고,
터널로 https 주소를 만들어야 한다. 발표 전에 반드시 실제 기기로 한 번 확인할 것.

왜 판정을 따로 떼어놨나 — webrtc는 프레임을 별도 스레드로 준다. 거기서는 st.*를 부를 수 없고
(Streamlit은 스크립트 스레드에만 붙는다), 0.3초짜리 추론을 거기서 돌리면 그만큼 영상이 멈춘다.
그래서 이 파일이 하는 일은 둘뿐이다:
  - 받은 프레임은 손대지 않고 그대로 돌려준다  → 영상은 카메라 속도대로 부드럽게 흐른다
  - 판정은 작업 스레드가 '가장 최근 프레임' 하나만 집어서 돌리고 결과를 잠금 뒤에 놓는다
영상 프레임 수와 판정 횟수를 분리하는 것이 핵심이다. CPU에서 초당 3번 판정해도 영상은 30fps다.
쌓인 프레임을 버리는 이유도 같다 — 다음 판정은 더 새로운 장면으로 하는 것이 맞다.
"""
import inspect
import threading
import time

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    AVAILABLE = True
except Exception:          # 패키지 없음, 또는 av/aiortc 설치 실패
    WebRtcMode = webrtc_streamer = None
    AVAILABLE = False

# ICE 설정 — 브라우저와 파이썬이 서로의 주소를 찾는 방법.
#
# 같은 컴퓨터(localhost)나 같은 공유기 안이면 서로의 주소를 이미 알기 때문에 아무것도 필요 없다.
# 이때 STUN을 켜두면 오히려 해롭다: 구글 STUN에 물어보고 답을 기다리는데, 방화벽이나 사내망이
# 그 UDP를 막으면 응답이 안 와서 "Connection is taking longer than expected"로 몇 초씩 매달린다.
# 필요 없는 의존을 기본값에서 뺀다.
#
# STUN이 실제로 필요한 경우는 둘 사이에 NAT가 낀 진짜 원격 접속이다. 그때만 켠다.
RTC_CONFIG_LOCAL = {"iceServers": []}
RTC_CONFIG_STUN = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


MIN_INTERVAL = 1.0    # 판정 사이 최소 간격(초). 0이면 쉬지 않고 연달아 돈다 — 그러면 영상이 죽는다

# 카메라 해상도. 어차피 224로 줄여 넣으므로 320이면 모델에 손해가 없다. 해상도를 올리면
# 브라우저에서 받아 디코딩하고 다시 인코딩해 돌려보내는 비용만 제곱으로 는다.
QUALITY = {"낮음 (320p · 가장 안정적)": 320, "보통 (480p)": 480, "높음 (640p)": 640}


class LiveJudge:
    """webrtc 프레임을 받아 영상은 그대로 통과시키고, 뒤에서 최신 프레임만 판정한다.

    min_interval이 왜 필요한가 — 판정을 쉬지 않고 돌리면 CPU를 다 먹는다. 그런데 영상을 받아
    디코딩하고 연결을 유지하는 일(aiortc)도 같은 CPU에서 돌아간다. 추론이 코어를 독점하면 그쪽이
    굶어서 영상이 점점 느려지다가 연결이 끊긴다 — 실제로 6초 만에 끊겼다. 판정을 초당 두 번으로
    묶어두면 남는 시간이 영상 쪽으로 간다. 판정이 촘촘할 필요도 없다: 사람이 건축물을 비추고 있는
    동안 초당 두 번이면 충분하고, 그 사이 영상은 30fps로 흐른다.
    """

    def __init__(self, predict, min_interval: float = MIN_INTERVAL):
        self._predict = predict           # (image_bgr) -> {"pred":…, "infer_ms":…, "error":…}
        self.min_interval = min_interval  # 화면에서 바꿀 수 있게 공개 속성으로 둔다
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending = None              # 아직 판정 안 한 최신 프레임 (덮어쓴다)
        self._latest = None               # 마지막으로 변환한 프레임. capture()가 이걸 준다
        self._want = True                 # 작업 스레드가 다음 한 장을 기다리는 중인가
        self._result = None
        self._frames = 0                  # 카메라가 보낸 장수
        self._judged = 0                  # 실제로 판정한 장수
        self._stop = False
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # ---- webrtc 프레임 스레드가 부른다. 무거운 일을 하면 안 되는 자리 ----
    def recv(self, frame):
        """영상 경로에서 불린다. 여기서 시간을 쓰면 연결 유지 신호가 밀려 통째로 끊긴다.

        to_ndarray는 색공간 변환이라 공짜가 아니다. 초당 15장을 전부 변환해놓고 실제로는 한두 장만
        판정에 쓰는 건 낭비다. 그래서 작업 스레드가 '다음 장을 기다리는 중'일 때만 변환한다.
        """
        with self._lock:
            self._frames += 1
            if not self._want:
                return frame              # 판정할 때가 아니면 변환조차 하지 않는다
        image = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._pending = image
            self._latest = image
            self._want = False
            self._wake.notify()
        return frame                      # 손대지 않고 그대로 돌려준다

    def capture(self):
        """지금 카메라 앞 장면 한 장(복사본). 아직 프레임이 없으면 None.

        실시간 판정은 계속 바뀐다. 한 장을 멈춰 세워 판정과 해설을 차분히 보고 싶을 때 쓴다.
        """
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    # ---- 판정 작업 스레드 ----
    def _loop(self):
        while True:
            with self._lock:
                self._want = True          # 이제 한 장 받을 준비가 됐다 — recv가 이때만 변환한다
                while self._pending is None and not self._stop:
                    self._wake.wait(0.5)
                if self._stop:
                    return
                image, self._pending = self._pending, None
            # 잠금 밖에서. 여기가 0.3초짜리다.
            # 예외를 삼키는 이유: 이 스레드가 죽으면 화면은 마지막 판정을 그대로 물고 멈춘다 —
            # 틀린 결과를 계속 보여주면서 아무 표시도 없는 것이 제일 나쁘다. 화면에 실패로 띄운다.
            started = time.perf_counter()
            try:
                out = self._predict(image)
            except Exception as e:
                out = {"pred": None, "infer_ms": None, "error": f"{type(e).__name__}: {e}"}
            with self._lock:
                self._result = out
                self._judged += 1
            rest = self.min_interval - (time.perf_counter() - started)
            if rest > 0:
                time.sleep(rest)      # 이 틈이 영상 쪽 몫이다. 없으면 연결이 끊긴다

    # ---- 화면(스크립트 스레드)이 읽는다 ----
    def snapshot(self) -> tuple[dict | None, int, int]:
        """(마지막 판정 결과, 받은 프레임 수, 판정한 수). 아직 한 번도 안 끝났으면 결과는 None."""
        with self._lock:
            return self._result, self._frames, self._judged

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._wake.notify_all()

    # streamlit-webrtc가 정리할 때 부른다
    on_ended = stop


def stream(judge: LiveJudge, key: str = "live", width: int = 320, use_stun: bool = False,
           fps: int = 10):
    """이미 만들어둔 judge로 webrtc 스트림을 띄우고 context를 돌려준다.

    judge를 밖에서 받는 이유 — streamlit-webrtc는 버전마다 프레임을 넘기는 방법이 다르다
    (video_frame_callback / video_processor_factory). 어느 쪽이든 우리는 같은 judge 하나를
    쓰고 결과도 거기서 직접 읽는다. ctx.video_processor에 기대지 않으므로 버전이 바뀌어도
    화면 코드는 그대로다. 어떤 인자를 받는지는 실제 함수 서명을 보고 고른다.
    """
    if webrtc_streamer is None:
        raise RuntimeError("streamlit-webrtc가 설치되지 않았다")
    kwargs = {
        "key": key,
        "mode": WebRtcMode.SENDRECV,
        "rtc_configuration": RTC_CONFIG_STUN if use_stun else RTC_CONFIG_LOCAL,
        # 해상도와 fps를 낮게 잡는 이유: 받은 영상을 디코딩하고 다시 인코딩해 브라우저로 돌려주는
        # 일이 전부 같은 파이썬 프로세스에서 돈다. 여기가 밀리면 연결 유지 신호가 늦어 통째로 끊긴다.
        # 판정 정확도에는 영향이 없다 — 어차피 224로 줄여서 모델에 넣는다.
        "media_stream_constraints": {
            "video": {"width": {"ideal": width}, "frameRate": {"ideal": fps}},
            "audio": False,
        },
    }
    params = inspect.signature(webrtc_streamer).parameters
    if "video_frame_callback" in params:          # 요즘 방식: 프레임마다 부르는 함수 하나
        kwargs["video_frame_callback"] = judge.recv
    elif "video_processor_factory" in params:     # 예전 방식: recv()를 가진 객체를 만드는 함수
        kwargs["video_processor_factory"] = lambda: judge
    else:
        raise RuntimeError(
            "이 streamlit-webrtc 버전에 프레임을 넘길 방법을 못 찾았다 "
            f"(받는 인자: {sorted(params)}). 버전을 확인해야 한다")
    if "async_processing" in params:
        kwargs["async_processing"] = True         # 프레임 처리를 별도로 돌린다
    return webrtc_streamer(**{k: v for k, v in kwargs.items() if k in params})
