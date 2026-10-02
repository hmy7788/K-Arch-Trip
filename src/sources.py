"""입력 소스: 사진 폴더와 웹캠. 웹캠은 실제로 열리는지 확인해서 없으면 목록에서 뺀다."""
import sys
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

FOLDER, WEBCAM = "폴더", "웹캠"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CAPTURE_SIZE = (640, 480)   # 어차피 224로 줄여 넣는다. 더 키워봐야 정확도는 그대로고 지연만 는다

try:  # 웹캠이 없을 때 OpenCV가 찍는 경고 줄이기
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def _open_camera(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # 윈도우 기본(MSMF)은 여는 데 수 초
    return cv2.VideoCapture(index, backend)


def webcam_available(index: int = 0) -> bool:
    cap = _open_camera(index)
    try:
        return bool(cap.isOpened() and cap.read()[0])
    finally:
        cap.release()


def first_camera(max_index: int = 3) -> int | None:
    """열리는 첫 카메라 번호. 노트북에 Windows Hello 적외선 카메라가 있으면 0번이 그쪽일 수 있다."""
    for i in range(max_index + 1):
        if webcam_available(i):
            return i
    return None


def available_sources(probe: Callable[[], bool] = webcam_available) -> list[str]:
    """폴더는 항상, 웹캠은 probe()가 True일 때만."""
    return [FOLDER, WEBCAM] if probe() else [FOLDER]


def list_images(folder: Path | str) -> list[str]:
    folder = Path(folder)
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_image(path: Path | str) -> np.ndarray:
    """cv2.imread와 같은 BGR uint8. 윈도우 한글 경로도 되게 imdecode."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path}: 이미지로 디코딩 실패")
    return image


class _Freshest:
    """카메라를 별도 스레드로 계속 읽어 '가장 최근 프레임'만 남긴다.

    한 장 판정에 0.3초가 걸리면 그 사이 카메라는 10장쯤 더 찍는다. 그 장면들이 드라이버 버퍼에
    쌓이면 화면은 몇 초 전 장면을 보여준다 — 느린 게 아니라 늦는 것이고, 손을 움직여도 화면이
    따라오지 않는 이유가 이거다. 쌓인 것을 버리고 최신 것만 내주면 판정은 항상 '지금 카메라 앞'에
    대해 내려진다. 버린 프레임은 어차피 판정에 안 쓴다.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._frame: np.ndarray | None = None
        self._seq = 0                       # 프레임 번호. 같은 장을 두 번 내주지 않으려고 센다
        self._dead = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._cond:
                if self._dead:
                    return
            ok, frame = self._cap.read()        # 잠금 밖에서. 다음 프레임까지 블록된다
            with self._cond:
                if self._dead:
                    return
                if not ok:
                    self._dead = True
                    self._cond.notify_all()
                    return
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def next(self, last_seq: int, timeout: float = 3.0) -> tuple[int, np.ndarray | None]:
        """last_seq 다음에 들어온 프레임. 카메라가 끊기거나 timeout이면 프레임 자리에 None."""
        with self._cond:
            self._cond.wait_for(lambda: self._dead or self._seq > last_seq, timeout)
            if self._dead or self._seq <= last_seq:
                return self._seq, None
            return self._seq, self._frame

    def close(self) -> None:
        with self._cond:
            self._dead = True
            self._cond.notify_all()
        self._thread.join(timeout=1.0)   # read() 도중에 release하면 백엔드가 죽는 수가 있다
        self._cap.release()


def webcam_frames(index: int = 0, size: tuple[int, int] | None = CAPTURE_SIZE
                  ) -> Iterator[tuple[str, np.ndarray]]:
    """읽기가 실패할 때까지, 항상 가장 최근 프레임만. 소비를 멈추면 close()로 카메라를 놓는다."""
    cap = _open_camera(index)
    if not cap.isOpened():
        cap.release()
        return
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 먹는 백엔드에서만 듣는다. 진짜 방어는 _Freshest 쪽
    reader = _Freshest(cap)
    try:
        seq = 0
        while True:
            seq, frame = reader.next(seq)
            if frame is None:
                return
            yield f"webcam/{datetime.now():%Y%m%d_%H%M%S_%f}.jpg", frame
    finally:
        reader.close()
