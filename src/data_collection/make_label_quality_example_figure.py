"""라벨 품질 필터링(강한 신호로 제외된 사진)의 실제 예시를 발표자료용 그림으로 만든다.

label_quality_report.md에서 "강한 신호"(룰베이스+Mask R-CNN 둘 다 폴더 라벨과
다르게 판정)로 제외된 사진 중, 두 트랙이 서로 같은 판정으로 일치하는(=우연이
아니라 진짜 이상한 사진일 가능성이 높은) 사례를 골라 원본 사진 + 폴더 라벨/두
트랙 판정을 나란히 보여준다.

실행:
    python src/data_collection/make_label_quality_example_figure.py
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

# (raw2 내 경로, 폴더 라벨, 룰베이스 판정, Mask R-CNN 판정) — label_quality_report.md에서
# 두 트랙이 서로 일치하는 "강한 신호" 사례만 선별(우연이 아닐 가능성이 높음).
EXAMPLES = [
    ("mug/180.jpg", "mug", "straight", "straight"),
    ("taper_step/78.jpg", "taper_step", "straight", "straight"),
    ("taper_step/64.jpg", "taper_step", "mug", "mug"),
    ("straight/173.jpg", "straight", "mug", "mug"),
]

SHAPE_LABELS_KO = {
    "mug": "머그형",
    "straight": "직선 원통형",
    "taper_smooth": "연속 테이퍼형",
    "taper_step": "단차 테이퍼형",
}


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw2-root", default="data/raw2")
    parser.add_argument("--out", default="reports/figures/data_pipeline/label_quality_strong_signal_examples.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, len(EXAMPLES), figsize=(4 * len(EXAMPLES), 5))
    fig.suptitle("라벨 품질 필터링 — \"강한 신호\"(제외) 실제 예시", fontsize=15, fontweight="bold")

    for ax, (rel_path, folder_label, rule_pred, maskrcnn_pred) in zip(axes, EXAMPLES):
        path = os.path.join(args.raw2_root, rel_path)
        print(f"처리 중: {path}")
        image_bgr = imread_unicode(path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        ax.axis("off")
        title = (f"폴더 라벨: {SHAPE_LABELS_KO[folder_label]}\n"
                 f"룰베이스: {SHAPE_LABELS_KO[rule_pred]}\n"
                 f"Mask R-CNN: {SHAPE_LABELS_KO[maskrcnn_pred]}\n"
                 f"→ 강한 신호(제외)")
        ax.set_title(title, fontsize=11, color="#B00020")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
