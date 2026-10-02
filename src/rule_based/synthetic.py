"""테스트/데모용 합성 실루엣 마스크 생성 헬퍼.

실제 텀블러 사진이 없는 단계에서, classify_shape()의 판정 로직 자체를 검증하기
위한 4종 형태의 깔끔한 합성 마스크를 만든다.
"""

import cv2
import numpy as np

MARGIN = 20


def _blank_canvas(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _add_handle_bump(mask: np.ndarray, right_edge_x: int, y_center: int,
                      protrusion: int = 40, height: int = 50) -> np.ndarray:
    """몸통 오른쪽에 사각형 돌출부(손잡이)를 이어 붙여 비대칭 돌출을 만든다."""
    x1 = max(right_edge_x - 5, 0)  # 몸통과 겹치게 시작해 확실히 이어붙인다
    x2 = right_edge_x + protrusion
    y1 = y_center - height // 2
    y2 = y_center + height // 2
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def make_mug_mask(with_handle: bool = False) -> np.ndarray:
    """높이 60 / 지름 140 (비율 60/140 ≈ 0.43, mug 조건인 <=1.5 를 여유 있게 만족)."""
    height, width = 60, 140
    canvas = _blank_canvas(height + 2 * MARGIN, width + 2 * MARGIN + (60 if with_handle else 0))
    x1, y1 = MARGIN, MARGIN
    x2, y2 = MARGIN + width, MARGIN + height
    cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, thickness=-1)
    if with_handle:
        canvas = _add_handle_bump(canvas, x2, (y1 + y2) // 2, protrusion=40, height=height // 2)
    return canvas


def make_straight_mask(with_handle: bool = False) -> np.ndarray:
    """높이 300 / 위아래 폭 동일(100) -> bottom/top 비율 1.0."""
    height, width = 300, 100
    canvas = _blank_canvas(height + 2 * MARGIN, width + 2 * MARGIN + (60 if with_handle else 0))
    x1, y1 = MARGIN, MARGIN
    x2, y2 = MARGIN + width, MARGIN + height
    cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, thickness=-1)
    if with_handle:
        canvas = _add_handle_bump(canvas, x2, (y1 + y2) // 2, protrusion=40, height=50)
    return canvas


def make_taper_smooth_mask(with_handle: bool = False) -> np.ndarray:
    """높이 300, 위폭 100 -> 아래폭 40 으로 꺾임 없이 선형으로 좁아짐."""
    height, top_width, bottom_width = 300, 100, 40
    canvas_width = top_width + 2 * MARGIN + (60 if with_handle else 0)
    canvas = _blank_canvas(height + 2 * MARGIN, canvas_width)
    cx = MARGIN + top_width // 2
    y_top, y_bottom = MARGIN, MARGIN + height
    pts = np.array([
        [cx - top_width // 2, y_top],
        [cx + top_width // 2, y_top],
        [cx + bottom_width // 2, y_bottom],
        [cx - bottom_width // 2, y_bottom],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 255)
    if with_handle:
        # 구간 경계에 걸치지 않도록 1/4 높이 지점(한 구간 안)에 작게 배치한다 —
        # 경계에 걸치면 median smoothing으로도 안 지워지는 인공적인 step이
        # 생겨서(양쪽 구간이 서로 다른 값으로 스무딩됨) classify_shape()의
        # 실측 기반 STEP_JUMP_RATIO_THRESHOLD를 오탐하게 만든다.
        y_frac = 0.25
        y_pos = int(y_top + height * y_frac)
        right_edge = cx + (top_width // 2 + (bottom_width // 2 - top_width // 2) * y_frac)
        canvas = _add_handle_bump(canvas, int(right_edge), y_pos, protrusion=40, height=20)
    return canvas


def make_taper_step_mask(with_handle: bool = False) -> np.ndarray:
    """위쪽 절반은 폭 100, 아래쪽 절반은 폭 40 으로 중간에서 한 번 꺾임."""
    height, top_width, bottom_width = 300, 100, 40
    canvas_width = top_width + 2 * MARGIN + (60 if with_handle else 0)
    canvas = _blank_canvas(height + 2 * MARGIN, canvas_width)
    cx = MARGIN + top_width // 2
    y_top = MARGIN
    y_mid = MARGIN + height // 2
    y_bottom = MARGIN + height

    upper = np.array([
        [cx - top_width // 2, y_top], [cx + top_width // 2, y_top],
        [cx + top_width // 2, y_mid], [cx - top_width // 2, y_mid],
    ], dtype=np.int32)
    lower = np.array([
        [cx - bottom_width // 2, y_mid], [cx + bottom_width // 2, y_mid],
        [cx + bottom_width // 2, y_bottom], [cx - bottom_width // 2, y_bottom],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [upper, lower], 255)
    if with_handle:
        canvas = _add_handle_bump(canvas, cx + top_width // 2, y_top + height // 4, protrusion=40, height=50)
    return canvas
