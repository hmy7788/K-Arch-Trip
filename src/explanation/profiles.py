"""형태별 해설 로더 + 판정 규칙. class_profiles.yaml을 읽어 화면에 쓸 형태로 돌려준다.

Streamlit을 import하지 않는다 — 브라우저 없이 테스트된다.
"""
from pathlib import Path

import yaml

PROFILE_PATH = Path(__file__).parent / "class_profiles.yaml"
FIELDS = ("design", "movement", "products")
FIELD_LABELS = {"design": "형태의 구조적 이유",
                "movement": "건축 사조 · 배경",
                "products": "대표 건축물"}
UNDECIDED = "판정 보류"


def load_profiles(path: Path | None = None) -> dict[str, dict]:
    """{클래스: {title, ko, rule, features, design, movement, products}}.

    파일이 없으면 {} — 해설 없다고 판정이 멈추면 안 된다.
    features는 결과 카드의 '주요 특징' 칩. 리스트가 아니면 빈 리스트로 둔다(문구 하나 때문에 화면이 죽지 않게).
    """
    path = path or PROFILE_PATH
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        feats = val.get("features")
        out[str(key)] = {
            "title": str(val.get("title") or key),
            "ko": str(val.get("ko") or ""),
            "rule": str(val.get("rule") or ""),
            "features": [str(f).strip() for f in feats if str(f).strip()] if isinstance(feats, list) else [],
            **{f: " ".join(str(val.get(f) or "").split()) for f in FIELDS},
        }
    return out


def missing_classes(profiles: dict[str, dict], classes: list[str]) -> list[str]:
    return [c for c in classes if c not in profiles]


def headline(profiles: dict[str, dict], label: str) -> str:
    """'계단식 적층형 건축' 한 줄. 해설이 없으면 클래스 이름 그대로.

    한글을 앞에 둔다 — 도감에서 눈이 먼저 닿는 건 영문 양식명이 아니라 우리말 이름이다.
    영문명은 카드에서 부제로 따로 붙인다(demo.py).
    """
    p = profiles.get(label)
    if not p:
        return label
    return p["ko"] or p["title"]


def subtitle(profiles: dict[str, dict], label: str | None) -> str:
    """'Step Taper Architecture' — 카드에 한글 이름 아래로 붙는 영문 양식명."""
    p = profiles.get(label or "")
    return p["title"] if p else ""


def features(profiles: dict[str, dict], label: str | None) -> list[str]:
    """'주요 특징' 칩. 없으면 빈 리스트 — 칩 줄만 빠지고 나머지는 그대로 나온다."""
    p = profiles.get(label or "")
    return list(p["features"]) if p else []


def decide(profiles: dict[str, dict], label: str | None, score: float | None,
           threshold: float) -> tuple[str, str, str]:
    """(화면 제목, 설명 문구, 상태). 상태는 'ok' | 'undecided' | 'none'.

    모델 출력(label·score)과 우리가 내리는 판정은 다른 층이다. 확신도 하나로 가른다.
    임계값이 막는 것: 모델이 헷갈리는 사진. 못 막는 것: 물체가 너무 작아 볼 것이 없는 사진 —
    그땐 헷갈리지 않고 확신하며 틀린다. softmax 최댓값은 항상 1/클래스수 이상이라 그 아래로
    내리면 아무것도 걸러지지 않는다.
    """
    if score is None:
        return UNDECIDED, "이 모델은 확신도를 내주지 않는다", "none"
    if score < threshold:
        return UNDECIDED, f"확신도 {score:.2f} < 기준 {threshold:.2f}", "undecided"
    return headline(profiles, label), f"판정 완료 · 확신도 {score:.2f} ≥ 기준 {threshold:.2f}", "ok"


def top_candidates(probs: dict[str, float] | None, n: int = 2, profiles: dict | None = None) -> str:
    """'처마 확장형 건축 0.33, 계단식 적층형 건축 0.28' — 보류일 때 후보만 알려준다."""
    if not probs:
        return ""
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])[:n]
    name = (lambda k: headline(profiles, k)) if profiles else (lambda k: k)
    return ", ".join(f"{name(k)} {v:.2f}" for k, v in ranked)
