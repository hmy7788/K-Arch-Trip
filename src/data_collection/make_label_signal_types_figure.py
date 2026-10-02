"""라벨 품질 필터링의 3가지 신호 유형(강한/약한/문제없음)을 원본+두 트랙 마스크와 함께 보여준다.

data/preprocess/label_quality_report.md(estimate_label_quality.py의 결과물)를
파싱해서 신호 유형별 후보를 모으고, --seed로 각 유형에서 한 장씩 무작위로
뽑는다. 뽑힌 사진만 룰베이스(get_mask)와 Mask R-CNN(get_mask_maskrcnn) 마스크를
새로 계산한다(전체 700장을 다시 돌리지 않음 — 이미 계산된 리포트를 재사용).

label_quality_report.md의 정의:
    강한 신호  — 룰베이스 + Mask R-CNN 둘 다 폴더 라벨과 다르게 판정 -> 제외
    약한 신호  — 둘 중 하나만 다르게 판정 -> 애매한 케이스일 수 있어 포함 유지
    문제 없음  — 둘 다 폴더 라벨과 동일하게 판정 -> 포함

실행:
    python src/data_collection/make_label_signal_types_figure.py            # 매번 다른 조합
    python src/data_collection/make_label_signal_types_figure.py --seed 3   # 재현 가능한 조합
"""

import argparse
import os
import random
import re
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rule_based.shape_classifier import get_mask, SHAPE_LABELS_KO  # noqa: E402
from deep_learning.dl1_maskrcnn import get_mask_maskrcnn, load_model  # noqa: E402

CLASSES = ("straight", "taper_smooth", "taper_step", "mug")
SIGNAL_TYPES = ("강한 신호", "약한 신호", "문제 없음")
SIGNAL_TITLES = {
    "강한 신호": "강한 신호 (제외)",
    "약한 신호": "약한 신호 (포함 유지)",
    "문제 없음": "문제 없음 (포함)",
}
SIGNAL_COLORS = {
    "강한 신호": "#B00020",
    "약한 신호": "#B8860B",
    "문제 없음": "#1B5E20",
}

ROW_RE = re.compile(r"^\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(강한 신호|약한 신호|문제 없음)\s*\|\s*$")


def parse_report(report_path: str):
    """label_quality_report.md를 파싱해 클래스별 (파일, 룰베이스, MaskRCNN, 신호) 리스트를 만든다."""
    by_signal = {s: [] for s in SIGNAL_TYPES}
    current_cls = None
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            header = re.match(r"^### (\S+)\s*$", line.strip())
            if header:
                current_cls = header.group(1)
                continue
            m = ROW_RE.match(line.strip())
            if m and current_cls in CLASSES:
                filename, rule_pred, maskrcnn_pred, signal = m.groups()
                by_signal[signal].append((current_cls, filename, rule_pred, maskrcnn_pred))
    return by_signal


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw2-root", default="data/raw2")
    parser.add_argument("--report", default="data/preprocess/label_quality_report.md")
    parser.add_argument("--seed", type=int, default=None, help="생략 시 매번 다른 조합이 뽑힘")
    parser.add_argument("--reroll", choices=SIGNAL_TYPES, action="append", default=[],
                         help="이 유형만 --seed와 별개로 재추첨(다른 유형은 --seed로 뽑은 그대로 유지). 여러 번 지정 가능")
    parser.add_argument("--reroll-seed", type=int, default=None, help="--reroll 항목을 뽑을 시드(생략 시 매번 다르게 뽑힘)")
    parser.add_argument("--out", default="reports/figures/data_pipeline/label_signal_types_examples.png")
    args = parser.parse_args()

    by_signal = parse_report(args.report)
    for s in SIGNAL_TYPES:
        print(f"{s}: 후보 {len(by_signal[s])}장")
    if any(not by_signal[s] for s in SIGNAL_TYPES):
        print("일부 신호 유형에 후보가 없습니다 — report 경로를 확인하세요.")
        return

    # 항상 SIGNAL_TYPES 순서대로 rng.choice()를 호출해 상태를 소비한다 — --reroll로
    # 지정된 유형이라도 이 호출 자체는 건너뛰지 않아야, 재추첨 대상이 아닌 뒤 순서
    # 유형(예: 문제 없음)이 같은 --seed에서 이전과 동일하게 뽑힌다.
    rng = random.Random(args.seed)
    picks = {}
    for s in SIGNAL_TYPES:
        picks[s] = rng.choice(by_signal[s])

    if args.reroll:
        reroll_rng = random.Random(args.reroll_seed)
        for s in args.reroll:
            picks[s] = reroll_rng.choice(by_signal[s])
            print(f"[재추첨] {s} -> 별도 시드로 다시 뽑음")

    print("\nMask R-CNN 모델 로드 중...")
    mrcnn_model, device = load_model()
    print(f"로드 완료 (device={device})\n")

    fig, axes = plt.subplots(len(SIGNAL_TYPES), 3, figsize=(12, 4 * len(SIGNAL_TYPES)))

    for row, signal in enumerate(SIGNAL_TYPES):
        cls, filename, rule_pred, maskrcnn_pred = picks[signal]
        path = os.path.join(args.raw2_root, cls, filename)
        print(f"[{signal}] {path} (룰베이스={rule_pred}, MaskRCNN={maskrcnn_pred})")

        image_bgr = imread_unicode(path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rule_mask = get_mask(image_bgr)
        mrcnn_mask = get_mask_maskrcnn(image_bgr, mrcnn_model, device)
        if mrcnn_mask is None:
            mrcnn_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        color = SIGNAL_COLORS[signal]

        ax_img = axes[row, 0]
        ax_img.imshow(image_rgb)
        ax_img.axis("off")
        ax_img.set_title(
            f"[{SIGNAL_TITLES[signal]}]\n"
            f"폴더 라벨: {SHAPE_LABELS_KO[cls]}\n"
            f"룰베이스: {SHAPE_LABELS_KO.get(rule_pred, rule_pred)} / "
            f"Mask R-CNN: {SHAPE_LABELS_KO.get(maskrcnn_pred, maskrcnn_pred)}",
            fontsize=10, color=color, loc="left")

        ax_rmask = axes[row, 1]
        ax_rmask.imshow(rule_mask, cmap="gray")
        ax_rmask.axis("off")
        ax_rmask.set_title("룰베이스 마스크", fontsize=10)

        ax_mmask = axes[row, 2]
        ax_mmask.imshow(mrcnn_mask, cmap="gray")
        ax_mmask.axis("off")
        ax_mmask.set_title("Mask R-CNN 마스크", fontsize=10)

    fig.suptitle("라벨 품질 필터링 — 신호 유형별 예시 (원본 + 룰베이스/Mask R-CNN 마스크)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
