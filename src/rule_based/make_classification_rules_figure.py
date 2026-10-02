"""4가지 판정 결과(머그/직선/단차/연속) 각각을 대표하는 사진으로 판정 규칙을 설명하는 그림을 만든다.

클래스별로 data/preprocess에서 "룰베이스가 폴더 라벨과 똑같이(=정확히) 판정한"
사진을 하나씩 찾아, 원본 -> 마스크 -> 폭 프로파일(판정에 쓰인 실제 수치 포함)을
보여준다. 어떤 사진에서 어떤 규칙이 실제로 발동했는지 한 세트로 확인 가능.

실행:
    python src/rule_based/make_classification_rules_figure.py
    python src/rule_based/make_classification_rules_figure.py --seed 3   # 다른 예시로 재추첨
"""

import argparse
import os
import random
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(__file__))
from shape_classifier import (  # noqa: E402
    classify_shape, get_mask, SHAPE_LABELS_KO,
    MUG_HEIGHT_TO_DIAMETER_MAX, STRAIGHT_BOTTOM_TOP_RATIO_MIN, STEP_JUMP_RATIO_THRESHOLD,
)

CLASSES = ("mug", "straight", "taper_step", "taper_smooth")

RULE_TEXT = {
    "mug": f"규칙1: 높이/지름 ≤ {MUG_HEIGHT_TO_DIAMETER_MAX} → 머그형",
    "straight": f"규칙2: 아래폭/위폭 ≥ {STRAIGHT_BOTTOM_TOP_RATIO_MIN} → 직선 원통형",
    "taper_step": f"규칙3: 단차 비율 > {STEP_JUMP_RATIO_THRESHOLD} → 단차 테이퍼형",
    "taper_smooth": "규칙4: 나머지 → 연속 테이퍼형",
}


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_samples(root: str, cls: str):
    class_dir = os.path.join(root, cls)
    names = [n for n in sorted(os.listdir(class_dir))
              if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png")]
    return [os.path.join(class_dir, n) for n in names]


def find_correct_example(root: str, cls: str, rng: random.Random, max_tries: int = 40):
    """폴더 라벨과 룰베이스 판정이 일치하는 사진을 찾을 때까지(최대 max_tries장) 시도."""
    paths = list_samples(root, cls)
    rng.shuffle(paths)
    for path in paths[:max_tries]:
        image = imread_unicode(path)
        mask = get_mask(image)
        if not mask.any():
            continue
        result = classify_shape(mask)
        if result["shape"] == cls:
            return path, image, mask, result
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/preprocess")
    parser.add_argument("--seed", type=int, default=None, help="생략 시 매번 다른 예시가 뽑힘")
    parser.add_argument("--out", default="reports/figures/rule_based/classification_rules_examples.png")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(11, 4 * len(CLASSES)))
    fig.suptitle("룰베이스 판정 기준 — 클래스별 실제 예시", fontsize=16, fontweight="bold")

    for row, cls in enumerate(CLASSES):
        print(f"[{cls}] 정확히 분류된 예시 탐색 중...")
        found = find_correct_example(args.data_root, cls, rng)
        if found is None:
            print(f"  경고: {cls} 예시를 못 찾음(max_tries 내에서 정확히 분류된 사진 없음)")
            for col in range(3):
                axes[row, col].axis("off")
            continue
        path, image_bgr, mask, result = found
        print(f"  선택: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        ax_img = axes[row, 0]
        ax_img.imshow(image_rgb)
        ax_img.axis("off")
        ax_img.set_title(f"{SHAPE_LABELS_KO[cls]} ({cls})\n{os.path.basename(path)}", fontsize=11)

        ax_mask = axes[row, 1]
        ax_mask.imshow(mask, cmap="gray")
        ax_mask.axis("off")
        ax_mask.set_title("마스크\n(Otsu+Canny → GrabCut)", fontsize=11)

        ax_profile = axes[row, 2]
        profile = result["width_profile"]
        ax_profile.barh(range(len(profile)), profile, color="#4C72B0")
        ax_profile.invert_yaxis()
        ax_profile.set_yticks(range(len(profile)))
        ax_profile.set_yticklabels([f"구간{i}" for i in range(len(profile))], fontsize=8)
        ax_profile.set_xlabel("폭(px)")
        ax_profile.set_title("폭 프로파일 → 판정", fontsize=11)
        info = (f"h/d = {result['height_diameter_ratio']:.2f}\n"
                f"아래/위 폭비 = {result['bottom_top_ratio']:.2f}\n"
                f"단차 비율 = {result['step_ratio']:.2f}\n"
                f"{RULE_TEXT[cls]}")
        ax_profile.text(0.97, 0.03, info, transform=ax_profile.transAxes, ha="right", va="bottom",
                         fontsize=9, color="#1B5E20",
                         bbox=dict(boxstyle="round", fc="#E8F5E9", ec="#1B5E20"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
