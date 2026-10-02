"""라벨 품질 필터링 규칙을 "진리표 + 실제 사진" 한 장으로 보여준다.

룰베이스/Mask R-CNN 판정이 폴더 라벨과 일치하는지 조합별로 4가지 경우가
있고, 그 조합에 따라 신호 유형과 최종 판단(유지/버림)이 정해진다 —
이 규칙과 각 조합의 실제 사진 예시를 한 그림에 정리한다.

    룰베이스 일치 + MaskRCNN 일치 -> 문제 없음 -> 유지
    룰베이스 일치 + MaskRCNN 다름 -> 약한 신호 -> 유지
    룰베이스 다름 + MaskRCNN 일치 -> 약한 신호 -> 유지
    룰베이스 다름 + MaskRCNN 다름 -> 강한 신호 -> 버림 (data/preprocess에서 제외됨)

실행:
    python src/data_collection/make_label_quality_truth_table_figure.py
"""

import argparse
import os
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

# (소스 루트, 클래스, 파일명, 룰베이스 판정, MaskRCNN 판정, 신호, 판단)
# 강한 신호 예시는 data/preprocess에서 이미 제외된 파일이라 data/raw2에서 가져온다.
ROWS = [
    ("data/preprocess", "mug", "14.jpg", "mug", "mug", "문제 없음", "유지"),
    ("data/preprocess", "straight", "72.jpg", "mug", "straight", "약한 신호", "유지"),
    ("data/preprocess", "mug", "103.jpg", "mug", "taper_step", "약한 신호", "유지"),
    ("data/raw2", "straight", "49.jpg", "mug", "mug", "강한 신호", "버림"),
]

ROW_COLORS = {
    "문제 없음": "#1B5E20",
    "약한 신호": "#B8860B",
    "강한 신호": "#B00020",
}


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="reports/figures/data_pipeline/label_quality_truth_table.png")
    args = parser.parse_args()

    print("Mask R-CNN 모델 로드 중...")
    mrcnn_model, device = load_model()
    print(f"로드 완료 (device={device})\n")

    fig = plt.figure(figsize=(15, 4.3 * len(ROWS)))
    gs = fig.add_gridspec(len(ROWS), 4, width_ratios=[1.5, 1, 1, 1], wspace=0.15, hspace=0.5)
    fig.suptitle("라벨 품질 필터링 진리표 — 룰베이스 x Mask R-CNN 조합별 실제 예시", fontsize=16, fontweight="bold")

    for row, (root, cls, fname, rule_pred, mrcnn_pred, signal, decision) in enumerate(ROWS):
        path = os.path.join(root, cls, fname)
        print(f"처리 중: {path} (신호={signal}, 판단={decision})")
        image_bgr = imread_unicode(path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rule_mask = get_mask(image_bgr)
        mrcnn_mask = get_mask_maskrcnn(image_bgr, mrcnn_model, device)
        if mrcnn_mask is None:
            mrcnn_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        color = ROW_COLORS[signal]

        # 왼쪽: 표 텍스트 (폴더라벨 / 룰베이스 / MaskRCNN / 신호 / 판단)
        ax_text = fig.add_subplot(gs[row, 0])
        ax_text.axis("off")
        rule_mark = "일치" if rule_pred == cls else f"다름({SHAPE_LABELS_KO.get(rule_pred, rule_pred)})"
        mrcnn_mark = "일치" if mrcnn_pred == cls else f"다름({SHAPE_LABELS_KO.get(mrcnn_pred, mrcnn_pred)})"
        text = (
            f"폴더 라벨: {SHAPE_LABELS_KO[cls]}\n\n"
            f"룰베이스: {rule_mark}\n"
            f"Mask R-CNN: {mrcnn_mark}\n\n"
            f"신호: {signal}\n"
            f"판단: {decision}"
        )
        ax_text.text(0.05, 0.5, text, transform=ax_text.transAxes, ha="left", va="center",
                     fontsize=13, color=color, fontweight="bold",
                     bbox=dict(boxstyle="round", fc="#FFF8E7" if signal != "강한 신호" else "#FDECEA", ec=color, lw=1.5))

        ax_img = fig.add_subplot(gs[row, 1])
        ax_img.imshow(image_rgb)
        ax_img.axis("off")
        ax_img.set_title("원본", fontsize=10)

        ax_rmask = fig.add_subplot(gs[row, 2])
        ax_rmask.imshow(rule_mask, cmap="gray")
        ax_rmask.axis("off")
        ax_rmask.set_title("룰베이스 마스크", fontsize=10)

        ax_mmask = fig.add_subplot(gs[row, 3])
        ax_mmask.imshow(mrcnn_mask, cmap="gray")
        ax_mmask.axis("off")
        ax_mmask.set_title("Mask R-CNN 마스크", fontsize=10)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
