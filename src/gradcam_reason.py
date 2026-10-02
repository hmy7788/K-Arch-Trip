"""GradCAM 히트맵의 공간 분포를 읽어 '왜 그렇게 봤는지' 문장을 만든다.

원본: 팀원이 준 gradcam_reason.py. 문장과 판정 로직은 그대로 두었다.

하는 일 — 히트맵을 상/중/하 × 엣지/중앙 6구역으로 나눠 평균 활성도를 재고, 활성도의 무게중심과
집중도를 구한 뒤, 클래스마다 "어느 구역이 높으면 어떤 근거" 규칙으로 문장을 고른다.

읽을 때 주의 — 이 문장들은 히트맵이 '어디를 봤는가'를 사람 말로 옮긴 것이지, 판정이 맞다는 증거가
아니다. 틀린 판정에도 그럴듯한 문장이 나온다. 근거를 읽고 이상하면(배경을 봤다든지) 그게 신호다.
"""
from __future__ import annotations

import numpy as np


# ── 공간 분석 ──────────────────────────────────────────────────────────────────

def _region_scores(cam: np.ndarray) -> dict[str, float]:
    """
    히트맵을 6개 구역으로 분할해 평균 활성도를 반환.

    구역 이름:
        top_edge / top_center
        mid_edge / mid_center
        bot_edge / bot_center

    cam: (H, W) float32, 0~1 정규화 완료된 GradCAM 출력
    """
    H, W = cam.shape
    t, b = H // 3, 2 * H // 3          # 상단 경계, 하단 경계
    ew   = max(1, W // 6)              # 엣지 너비 (좌우 각각)

    zones: dict[str, np.ndarray] = {
        "top_edge":    np.concatenate([cam[:t, :ew].ravel(), cam[:t, W-ew:].ravel()]),
        "top_center":  cam[:t, ew:W-ew].ravel(),
        "mid_edge":    np.concatenate([cam[t:b, :ew].ravel(), cam[t:b, W-ew:].ravel()]),
        "mid_center":  cam[t:b, ew:W-ew].ravel(),
        "bot_edge":    np.concatenate([cam[b:, :ew].ravel(), cam[b:, W-ew:].ravel()]),
        "bot_center":  cam[b:, ew:W-ew].ravel(),
    }
    return {k: float(v.mean()) for k, v in zones.items()}


def _center_of_mass(cam: np.ndarray) -> tuple[float, float]:
    """활성도 가중 중심 (row비율 0~1, col비율 0~1). 히트맵이 어디를 보는지 한 점으로 요약."""
    H, W = cam.shape
    total = cam.sum() + 1e-8
    rows = np.arange(H).reshape(-1, 1)
    cols = np.arange(W).reshape(1, -1)
    cy = float((cam * rows).sum() / total) / H   # 0=상단, 1=하단
    cx = float((cam * cols).sum() / total) / W   # 0=좌, 1=우
    return cy, cx


def _concentration(cam: np.ndarray, top_pct: float = 0.2) -> float:
    """
    상위 top_pct 픽셀이 전체 활성도에서 차지하는 비율.
    1.0에 가까울수록 좁은 곳에 집중, 낮을수록 고르게 분산.
    """
    threshold = np.quantile(cam, 1 - top_pct)
    return float(cam[cam >= threshold].sum() / (cam.sum() + 1e-8))


# ── 클래스별 판정 근거 문장 ────────────────────────────────────────────────────

def _reason_straight(scores: dict, cy: float, cx: float, conc: float,
                     conf: float) -> list[str]:
    lines = []

    # 주목 구역
    top_act   = (scores["top_edge"] + scores["top_center"]) / 2
    bot_act   = (scores["bot_edge"] + scores["bot_center"]) / 2
    edge_act  = (scores["top_edge"] + scores["mid_edge"] + scores["bot_edge"]) / 3
    cent_act  = (scores["top_center"] + scores["mid_center"] + scores["bot_center"]) / 3

    if edge_act > cent_act * 1.2:
        lines.append(
            "📍 **외벽 실루엣 전체**에 고르게 집중 — 꼭대기부터 지면까지 폭이 변하지 않는 "
            "수직 윤곽선을 확인했습니다."
        )
    elif abs(top_act - bot_act) < 0.05:
        lines.append(
            "📍 **위아래 폭이 동일**한 영역에 집중 — 아래 폭이 위 폭의 95% 이상 "
            "유지되는 수직 외벽을 확인했습니다."
        )
    else:
        dominant = "상단" if cy < 0.4 else ("하단" if cy > 0.6 else "중앙")
        lines.append(
            f"📍 **{dominant} 폭** 영역에 집중 — 전체 실루엣에서 폭 변화가 "
            "거의 없는 수직 직선형으로 판단했습니다."
        )

    if conc < 0.5:
        lines.append("💡 활성도가 전체에 분산돼 있어 실루엣 전반이 직선임을 종합적으로 판단했습니다.")

    if conf >= 0.90:
        lines.append("✅ 확신도 90% 이상 — 곡선 테이퍼형·처마 확장형과 명확히 구분되는 수직 실루엣이 포착되었습니다.")
    elif conf < 0.70:
        lines.append("⚠️ 상부 폭이 약간 좁아 보여 곡선 테이퍼형과 혼동될 수 있습니다. 건물을 정면에서 다시 찍어 보세요.")

    return lines


def _reason_taper_smooth(scores: dict, cy: float, cx: float, conc: float,
                         conf: float) -> list[str]:
    lines = []

    edge_act = (scores["top_edge"] + scores["mid_edge"] + scores["bot_edge"]) / 3
    bot_act  = (scores["bot_edge"] + scores["bot_center"]) / 2
    top_act  = (scores["top_edge"] + scores["top_center"]) / 2

    if edge_act > 0.35:
        lines.append(
            "📍 **외곽 곡선 라인**에 집중 — 아래에서 위로 꺾임 없이 "
            "연속해 좁아지는 매끄러운 윤곽이 확인되었습니다."
        )
    elif bot_act > top_act * 1.15:
        lines.append(
            "📍 **상부 좁아지는 구간**에 집중 — 꼭대기로 갈수록 폭이 줄어드는 "
            "테이퍼 구조의 끝점에 주목했습니다."
        )
    else:
        lines.append(
            "📍 **외곽 곡선**에 집중 — 넓은 기단부터 좁은 상부까지 매끄럽게 "
            "이어지는 윤곽선이 인식되었습니다."
        )

    if conc > 0.6:
        lines.append("💡 좁아지는 구간이 뚜렷하게 감지되어 집중 활성화가 나타났습니다.")

    if conf >= 0.85:
        lines.append("✅ 확신도 높음 — 층마다의 단차 없이 연속해 좁아지는 형태가 명확히 구분되었습니다.")
    elif conf < 0.70:
        lines.append("⚠️ 수직 직선형 또는 계단식 적층형과 혼동 가능성 있음 — 곡률이 미세하거나 올려다본 각도가 심하면 정확도가 낮아집니다.")

    return lines


def _reason_taper_step(scores: dict, cy: float, cx: float, conc: float,
                       conf: float) -> list[str]:
    lines = []

    bot_act  = (scores["bot_edge"] + scores["bot_center"]) / 2
    mid_act  = (scores["mid_edge"] + scores["mid_center"]) / 2
    top_act  = (scores["top_edge"] + scores["top_center"]) / 2

    step_zone_act = (scores["mid_edge"] + scores["bot_edge"]) / 2  # 단차 경계선은 중-하단 엣지

    if step_zone_act > 0.40:
        lines.append(
            "📍 **층이 바뀌는 경계선(단차)**에 집중 — 넓은 아래층과 좁은 위층 "
            "사이의 급격한 꺾임점이 포착되었습니다."
        )
    elif bot_act > top_act * 1.2:
        lines.append(
            "📍 **기단부**에 집중 — 위층을 받치는 넓은 기단과 그 위로 "
            "좁아지는 단차 구조가 인식되었습니다."
        )
    elif mid_act > bot_act and mid_act > top_act:
        lines.append(
            "📍 **단차 경계(중간 층)**에 집중 — 위아래 폭이 바뀌는 "
            "꺾임점 주변에 주목했습니다."
        )
    else:
        lines.append(
            "📍 **하부 적층 구조**에 집중 — 뚜렷한 단차를 만드는 아래층 형태가 "
            "계단식 적층형의 핵심 근거로 인식되었습니다."
        )

    if conc > 0.65:
        lines.append("💡 활성도가 특정 구역에 집중되어 단차 경계선이 뚜렷하게 감지되었습니다.")

    if conf >= 0.85:
        lines.append("✅ 확신도 높음 — 1회 이상의 뚜렷한 꺾임이 명확히 포착되었습니다.")
    elif conf < 0.70:
        lines.append("⚠️ 곡선 테이퍼형과 혼동 가능 — 단차가 완만하거나 역광이면 층 경계가 흐릿하게 보입니다.")

    return lines


def _reason_mug(scores: dict, cy: float, cx: float, conc: float,
                conf: float) -> list[str]:
    lines = []

    # 처마 확장형: 처마가 몸체 밖으로 뻗어 나가 좌우로 넓은 실루엣이 된다.
    # 한쪽으로만 뻗은 처마(좌우 비대칭)는 지금 구역 분할로는 못 본다 — 엣지 점수가 좌+우 평균이라서.
    # 그래서 아래 규칙은 무게중심(cy)과 엣지/중앙 대비만 쓴다.

    # 높이 대비 폭 — cy가 낮으면(지붕 쪽에 집중) 낮고 넓은 형태로 본다
    if cy < 0.45:
        lines.append(
            "📍 **상단 지붕면**에 집중 — 높이 대비 폭이 큰 처마 확장형 특유의 "
            "비례(높이 ≤ 폭 × 1.5)가 인식되었습니다."
        )
    elif scores["mid_edge"] > scores["mid_center"] * 1.3:
        lines.append(
            "📍 **처마가 뻗어 나온 자리**에 집중 — 몸체 밖으로 돌출된 구조가 "
            "처마 확장형의 핵심 특징으로 감지되었습니다."
        )
    else:
        lines.append(
            "📍 **건물 전체 비례**에 집중 — 수직 직선형보다 높이가 낮고 "
            "폭이 넓은 비율이 인식되었습니다."
        )

    if conc > 0.60:
        lines.append("💡 특정 구역(처마 끝 또는 지붕면)에 활성도가 집중되어 돌출 구조가 명확히 감지되었습니다.")

    if conf >= 0.85:
        lines.append("✅ 확신도 높음 — 낮은 높이 대비 넓은 폭과 뻗어 나온 처마가 명확히 포착되었습니다.")
    elif conf < 0.70:
        lines.append("⚠️ 수직 직선형과 혼동 가능 — 처마가 프레임 밖으로 잘리거나 정면 각도가 아니면 높이:폭 비율 판단이 어렵습니다.")

    return lines


# ── 확신도 티어 설명 ──────────────────────────────────────────────────────────

def _confidence_tier(conf: float, threshold: float) -> str:
    if conf >= 0.95:
        return f"확신도 {conf:.0%} — 매우 강한 특징이 포착되었습니다."
    elif conf >= 0.85:
        return f"확신도 {conf:.0%} — 해당 클래스의 특징이 명확하게 인식되었습니다."
    elif conf >= threshold:
        return f"확신도 {conf:.0%} — 판정 기준({threshold:.0%}) 이상이지만 다른 클래스와의 차이가 크지 않습니다."
    elif conf >= 0.50:
        return f"확신도 {conf:.0%} — 기준({threshold:.0%}) 미달. 건물을 정면에서, 밝을 때 다시 찍어보세요."
    else:
        return f"확신도 {conf:.0%} — 매우 낮음. 건물 전체가 프레임에 들어오게 물러서서 찍어보세요."


# ── 퍼블릭 API ────────────────────────────────────────────────────────────────

_REASON_FN = {
    "straight":     _reason_straight,
    "taper_smooth": _reason_taper_smooth,
    "taper_step":   _reason_taper_step,
    "mug":          _reason_mug,
}


def explain(cam: np.ndarray, label: str, score: float,
            threshold: float = 0.70) -> dict:
    """
    GradCAM 히트맵을 분석해 판정 근거를 생성한다.

    Parameters
    ----------
    cam       : (H, W) float32, 0~1 정규화된 GradCAM 출력
    label     : 예측 클래스 키 (straight / taper_smooth / taper_step / mug)
    score     : 모델 확신도 (0~1)
    threshold : 판정 완료 기준 확신도

    Returns
    -------
    {
        "focus_region":   str   — "상단 외곽 라인", "중앙 단차 경계" 등 한 줄 요약
        "reason_lines":   list[str]  — 판정 근거 문장 (1~3개)
        "confidence_str": str   — 확신도 티어 설명
        "attention_pct":  float — 상위 20% 픽셀이 전체 활성도에서 차지하는 비율 (집중도)
    }
    """
    cam_f = cam.astype(np.float32)
    scores = _region_scores(cam_f)
    cy, cx = _center_of_mass(cam_f)
    conc   = _concentration(cam_f)

    # 집중 구역 한 줄 요약
    vert   = "상단" if cy < 0.38 else ("하단" if cy > 0.62 else "중앙")
    edge_a = (scores["top_edge"] + scores["mid_edge"] + scores["bot_edge"]) / 3
    cent_a = (scores["top_center"] + scores["mid_center"] + scores["bot_center"]) / 3
    horiz  = "외곽 라인" if edge_a > cent_a * 1.15 else "중앙 면적"
    focus_region = f"{vert} {horiz}"

    # 클래스별 근거 문장
    fn = _REASON_FN.get(label)
    reason_lines = fn(scores, cy, cx, conc, score) if fn else [
        f"📍 클래스 '{label}'에 대한 세부 근거 매핑이 아직 없습니다."
    ]

    return {
        "focus_region":   focus_region,
        "reason_lines":   reason_lines,
        "confidence_str": _confidence_tier(score, threshold),
        "attention_pct":  round(conc * 100, 1),
    }
