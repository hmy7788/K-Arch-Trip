"""OpenCV 룰베이스 트랙: 텀블러 형태(4종) 분류.

판정 순서(README "분류 체계" 기준, 반드시 이 순서):
1) 높이/지름 <= MUG_HEIGHT_TO_DIAMETER_MAX  -> mug
2) 아래폭/위폭 >= STRAIGHT_BOTTOM_TOP_RATIO_MIN (거의 수직) -> straight
3) 폭 감소가 한 구간에 몰리지 않고 고르게 분산 -> taper_smooth
4) 폭 감소가 한 구간에 뚜렷하게 몰림(꺾임) -> taper_step

손잡이 유무(detect_handle)는 판정에 전혀 관여하지 않는다 — "실루엣이 손잡이보다
우선한다"는 규칙을 코드 구조로 강제하기 위해, classify_shape()의 분기 조건에는
handle_detected를 절대 사용하지 않는다. 손잡이 정보는 설명 텍스트 등 부가 정보로만
반환한다.
"""

import cv2
import numpy as np

# 임계값 — 2026-09-15, data/raw2 426장(Mask R-CNN 마스크 기준) 실측으로 재조정.
# 각 갈림길을 이진분류로 놓고 정확도를 최대화하는 지점을 탐색해서 구함
# (notebooks/threshold_calibration.ipynb). MUG는 원래 값(1.5)이 이미 최적에
# 가까웠고, STRAIGHT/STEP은 조정 후 전체 정확도 84.5%->88.0% 개선 확인
# (taper_smooth는 소폭 하락하는 trade-off 있음 — docs/experiment-log.md 참고).
MUG_HEIGHT_TO_DIAMETER_MAX = 1.5
STRAIGHT_BOTTOM_TOP_RATIO_MIN = 0.92
STEP_JUMP_RATIO_THRESHOLD = 0.29
HANDLE_DEPTH_RATIO_THRESHOLD = 0.08
MAX_PROCESSING_DIM = 1200  # 이보다 긴 변을 가진 이미지는 처리 전에 축소한다

SHAPE_LABELS_KO = {
    "mug": "머그형",
    "straight": "직선 원통형",
    "taper_smooth": "연속 테이퍼형",
    "taper_step": "단차 테이퍼형",
}


def _rough_mask_otsu(image: np.ndarray) -> np.ndarray:
    """Otsu 임계값 + 최대 외곽 컨투어로 대략적인 전경 마스크를 만든다.

    배경이 단일 색이고 대비가 뚜렷할 때만 안정적으로 동작하는 초기 추정치일
    뿐이다 — 이 결과는 get_mask()에서 GrabCut의 시드로만 쓰인다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if border.mean() > 127:
        binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def _rough_mask_canny(image: np.ndarray) -> np.ndarray:
    """엣지(윤곽선) 기반으로 대략적인 전경 마스크를 만든다.

    Otsu는 명도 하나만 보므로, 물체와 배경이 둘 다 흰색/연한 색 계열이면
    (예: 흰 텀블러 + 흰 배경) 거의 구분을 못 한다 (실측 확인: 그런 사진에서
    Otsu 전경 비율이 3~6%까지 떨어짐 — docs/troubleshooting.md 참고). 반면
    명도가 거의 같아도 물체 경계에는 옅은 그림자·그라데이션으로 인한 엣지가
    남는 경우가 많아서, Canny 엣지를 닫아서(morphological close) 윤곽을
    잡으면 이런 저대비 사진에서 Otsu보다 훨씬 잘 잡힌다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 60)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def _grabcut_refine(image: np.ndarray, rough_mask: np.ndarray) -> np.ndarray:
    """Otsu 결과를 시드로 GrabCut을 돌려 마스크를 정제한다.

    Otsu는 명도(그레이스케일) 하나만 보지만 GrabCut은 색 분포(GMM)를 전경/배경
    각각 따로 학습하기 때문에, 그림자가 물체에 붙거나 물체·배경 명도가 비슷한
    경우(나무 바닥, 코르크 받침 등)에 더 안정적이다. 실패하면 원본 rough_mask로
    조용히 폴백한다.
    """
    if image.ndim != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = rough_mask.shape
    border = max(2, int(0.03 * min(h, w)))
    kernel = np.ones((15, 15), np.uint8)
    sure_fg = cv2.erode(rough_mask, kernel, iterations=1)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[rough_mask > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD
    # 이미지 테두리는 배경이 확실하다고 가정(촬영 프로토콜상 피사체가 프레임
    # 가장자리까지 닿지 않음) — GrabCut이 배경 색 모델을 학습할 시드를 준다.
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return rough_mask

    refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if not refined.any():
        return rough_mask
    return refined


def get_mask(image: np.ndarray, use_grabcut: bool = True) -> np.ndarray:
    """이미지에서 전경(텀블러) 이진 마스크를 추출한다.

    Otsu(명도 기반)와 Canny(엣지 기반) 두 방법으로 각각 대략적인 마스크를
    만들어 합친 뒤 GrabCut으로 정제한다. 두 방법의 실패 유형이 서로 달라서
    (Otsu는 저대비 배경에 약하고, Canny는 텍스처 없는 밋밋한 배경에 강함)
    합쳐두면 한쪽이 실패해도 다른 쪽이 시드를 보완해준다. 그래도 완벽하지
    않다 — 물체가 사진에 일부만 나오거나(클로즈업 컷) 배경이 복잡한 경우는
    여전히 실패할 수 있으니, 분류 결과가 이상하면 저장된 마스크를 먼저 눈으로
    확인할 것.

    긴 변이 MAX_PROCESSING_DIM을 넘는 이미지는 처리 전에 축소한다 — 폰카메라
    원본(4000x3000 등)에서 GrabCut이 30초 넘게 걸리는 걸 실측으로 확인함
    (docs/troubleshooting.md). classify_shape()는 마스크의 절대 픽셀 수가
    아니라 비율(높이/지름 등)만 쓰므로 축소해도 판정에 영향 없다. 반환되는
    마스크는 (축소했다면) 축소된 크기를 따른다 — 원본과 크기가 다를 수 있음.
    """
    h, w = image.shape[:2]
    longest_side = max(h, w)
    if longest_side > MAX_PROCESSING_DIM:
        scale = MAX_PROCESSING_DIM / longest_side
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    rough_otsu = _rough_mask_otsu(image)
    rough_canny = _rough_mask_canny(image)
    rough = cv2.bitwise_or(rough_otsu, rough_canny)
    if not rough.any():
        return rough

    mask = _grabcut_refine(image, rough) if use_grabcut else rough

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)
    return cleaned


def _row_max_run_width(row: np.ndarray) -> int:
    """한 행(가로 한 줄)에서 가장 긴 연속 전경 픽셀 구간의 폭을 구한다.

    손잡이가 몸통과 살짝 떨어져 있어(구멍 사이로 배경이 보이는 D자형 등) 같은
    행에 몸통·손잡이 두 덩어리가 따로 찍히는 경우, 이전 방식(min-max 전체 폭)은
    손잡이 바깥쪽 끝~몸통 반대쪽 끝까지를 폭으로 오인식했다 — 실측으로 확인:
    손잡이 달린 taper_step 사진에서 폭 프로파일이 전 구간 거의 동일한 값으로
    나와 height/diameter가 mug 임계값 밑으로 뭉개짐(2026-09-16, docs/troubleshooting.md).
    가장 긴 연속 구간만 쓰면 배경으로 끊긴 손잡이 쪽은 자동으로 제외된다.
    손잡이가 몸통에 맞닿아 있는 경우(합성 테스트 등)는 어차피 하나의 연속
    구간으로 합쳐지므로 이전 방식과 동일하게 동작한다 — "실루엣이 손잡이보다
    우선" 원칙은 그대로 유지됨.
    """
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return 0
    gaps = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps, [idx.size - 1]))
    return int((idx[ends] - idx[starts] + 1).max())


def compute_width_profile(mask: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """마스크를 세로로 n_bins등분해 구간별 폭을 반환한다.

    index 0 = 몸통 위쪽(입구) 방향, index -1 = 아래쪽(바닥) 방향. 구간 폭은 그
    구간에 속한 행들의 "가장 긴 연속 전경 픽셀 구간" 폭의 중앙값이다(전체
    min-max 스팬이 아님) — 이유는 _row_max_run_width 참고.
    """
    ys, _ = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("mask에 전경 픽셀이 없습니다")

    y_min, y_max = ys.min(), ys.max()
    edges = np.linspace(y_min, y_max + 1, n_bins + 1)
    widths = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        y_start, y_end = int(edges[i]), int(edges[i + 1])
        row_widths = [_row_max_run_width(mask[y]) for y in range(y_start, y_end) if mask[y].any()]
        widths[i] = float(np.median(row_widths)) if row_widths else 0.0
    return widths


def detect_handle(mask: np.ndarray, depth_ratio_threshold: float = HANDLE_DEPTH_RATIO_THRESHOLD) -> bool:
    """convexHull + convexityDefects로 비대칭 돌출부(손잡이)를 검출한다.

    알려진 한계: 단차 테이퍼형(taper_step)의 꺾임 지점도 깊은 concavity를 만들어
    True로 잡힐 수 있다 (실제 손잡이와 기하학적으로 구분이 안 됨). classify_shape()는
    이 값을 판정에 쓰지 않으므로 분류 결과에는 영향 없음 — 설명 텍스트 등 부가
    정보로 쓸 때만 주의. 실측 데이터로 재조정 필요.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        return False

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return False
    hull_indices = np.sort(hull_indices, axis=0)

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return False
    if defects is None:
        return False

    _, _, w, h = cv2.boundingRect(contour)
    scale = max(w, h)
    if scale == 0:
        return False

    for defect in defects.reshape(-1, 4):
        depth_fixed_point = defect[3]
        depth = depth_fixed_point / 256.0
        if depth / scale >= depth_ratio_threshold:
            return True
    return False


def _median_smooth(profile: np.ndarray, window: int = 5) -> np.ndarray:
    """폭 프로파일에 1D median filter를 적용해, 손잡이 돌출부처럼 한 구간만 튀는
    값을 억제한다 (다중 구간에 걸친 진짜 단차는 그대로 유지됨). 손잡이 존재
    여부로 분기하는 게 아니라, 몸통 실루엣 측정 자체를 손잡이에 견고하게
    만드는 방식이라 "실루엣이 손잡이보다 우선" 원칙과 어긋나지 않는다.
    """
    pad = window // 2
    padded = np.pad(profile, pad, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(profile))])


def classify_shape(mask: np.ndarray, n_bins: int = 10) -> dict:
    """폭 프로파일만으로 형태(4종)를 판정한다. handle_detected는 참고 정보일 뿐 판정에 쓰이지 않는다."""
    profile = compute_width_profile(mask, n_bins=n_bins)
    smoothed = _median_smooth(profile)

    ys, _ = np.nonzero(mask)
    height = float(ys.max() - ys.min() + 1)
    diameter = float(smoothed.max())
    height_diameter_ratio = height / diameter if diameter > 0 else float("inf")

    top_width = float(smoothed[0])
    bottom_width = float(smoothed[-1])
    bottom_top_ratio = bottom_width / top_width if top_width > 0 else 0.0

    handle_detected = detect_handle(mask)

    # 항상 계산해서 반환한다 — taper 분기가 아니어도(mug/straight로 갈려도)
    # step_ratio 값 자체는 임계값 재조정(캘리브레이션) 등에 쓸 수 있어야 하므로.
    deltas = smoothed[:-1] - smoothed[1:]  # 양수 = 아래로 가며 폭이 줄어듦
    total_drop = float(smoothed[0] - smoothed[-1])
    step_ratio = float(deltas.max() / total_drop) if total_drop > 0 else 0.0

    if height_diameter_ratio <= MUG_HEIGHT_TO_DIAMETER_MAX:
        shape = "mug"
    elif bottom_top_ratio >= STRAIGHT_BOTTOM_TOP_RATIO_MIN:
        shape = "straight"
    else:
        shape = "taper_step" if step_ratio > STEP_JUMP_RATIO_THRESHOLD else "taper_smooth"

    return {
        "shape": shape,
        "height_diameter_ratio": height_diameter_ratio,
        "bottom_top_ratio": bottom_top_ratio,
        "step_ratio": step_ratio,
        "width_profile": profile.tolist(),
        "handle_detected": handle_detected,
    }


def describe_result_ko(result: dict) -> str:
    """classify_shape()의 반환값을 사람이 읽기 좋은 한글 설명으로 바꾼다."""
    shape_ko = SHAPE_LABELS_KO.get(result["shape"], result["shape"])
    profile_str = ", ".join(f"{w:.1f}" for w in result["width_profile"])
    handle_ko = "예" if result["handle_detected"] else "아니오"
    return (
        f"형태            : {shape_ko} ({result['shape']})\n"
        f"높이/지름 비율   : {result['height_diameter_ratio']:.2f}\n"
        f"아래폭/위폭 비율 : {result['bottom_top_ratio']:.2f}\n"
        f"손잡이 감지      : {handle_ko}\n"
        f"폭 프로파일(위→아래) : [{profile_str}]"
    )
