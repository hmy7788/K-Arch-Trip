"""룰베이스 트랙을 한 장으로 설명하는 발표자료용 그림을 만든다.

위쪽: 파이프라인 예시 한 장(원본 -> 마스크 -> 폭 프로파일과 판정 근거).
아래쪽: data/test1 confusion matrix + 핵심 지표(정확도, Macro-F1) + 한줄 결론.

실행:
    python src/rule_based/make_ppt_figure.py
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

sys.path.insert(0, os.path.dirname(__file__))
from shape_classifier import classify_shape, get_mask, SHAPE_LABELS_KO  # noqa: E402

CLASSES = ("straight", "taper_smooth", "taper_step", "mug")


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_samples(root: str):
    samples = []
    for cls in CLASSES:
        class_dir = os.path.join(root, cls)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png"):
                samples.append((os.path.join(class_dir, name), cls))
    return samples


def evaluate(samples):
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    correct = 0
    for path, cls in samples:
        image = imread_unicode(path)
        mask = get_mask(image)
        pred = classify_shape(mask)["shape"] if mask.any() else "none"
        if pred in cls_to_idx:
            matrix[cls_to_idx[cls], cls_to_idx[pred]] += 1
        correct += int(pred == cls)
    return matrix, correct / len(samples)


def macro_f1_of(matrix) -> float:
    f1s = []
    for i in range(len(CLASSES)):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    return sum(f1s) / len(f1s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-root", default="data/test1")
    parser.add_argument("--example", default="data/test1/straight/KakaoTalk_20260915_114237129_04.jpg",
                         help="파이프라인 예시로 보여줄 사진 (정확히 분류된 사진을 고르는 걸 권장)")
    parser.add_argument("--out", default="reports/figures/rule_based/ppt_summary.png")
    args = parser.parse_args()

    print(f"예시 이미지 처리 중: {args.example}")
    image_bgr = imread_unicode(args.example)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = get_mask(image_bgr)
    result = classify_shape(mask)

    print(f"{args.test_root} 전체 평가 중...")
    samples = list_samples(args.test_root)
    matrix, accuracy = evaluate(samples)
    macro_f1 = macro_f1_of(matrix)
    print(f"정확도 {accuracy*100:.1f}%, Macro-F1 {macro_f1:.3f}")

    fig = plt.figure(figsize=(13, 9), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.38, wspace=0.32,
                           top=0.90, bottom=0.07, left=0.06, right=0.97)

    fig.suptitle("룰베이스 트랙 — OpenCV Width Profile (get_mask + classify_shape)", fontsize=18, fontweight="bold")

    # 1) 원본
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image_rgb)
    ax.set_title("① 원본 사진", fontsize=13)
    ax.axis("off")

    # 2) 마스크
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mask, cmap="gray")
    ax.set_title("② 전경 마스크\n(Otsu+Canny → GrabCut)", fontsize=13)
    ax.axis("off")

    # 3) 폭 프로파일 + 판정
    ax = fig.add_subplot(gs[0, 2])
    profile = result["width_profile"]
    bars = ax.barh(range(len(profile)), profile, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_yticks(range(len(profile)))
    ax.set_yticklabels([f"구간{i}" for i in range(len(profile))], fontsize=8)
    ax.set_xlabel("폭(px)")
    ax.set_title("③ 폭 프로파일 → 판정 근거", fontsize=13)
    pred_ko = SHAPE_LABELS_KO[result["shape"]]
    info = (f"h/d = {result['height_diameter_ratio']:.2f}\n"
            f"아래/위 폭비 = {result['bottom_top_ratio']:.2f}\n"
            f"판정: {pred_ko}({result['shape']})")
    ax.text(0.97, 0.03, info, transform=ax.transAxes, ha="right", va="bottom",
             fontsize=10, bbox=dict(boxstyle="round", fc="#FFF6D9", ec="#C9A227"))

    # 4) confusion matrix
    ax = fig.add_subplot(gs[1, 0:2])
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("예측"); ax.set_ylabel("실제")
    ax.set_title(f"④ Test Confusion Matrix ({args.test_root}, {len(samples)}장)", fontsize=13)
    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if matrix[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 5) 핵심 지표 + 결론
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    summary = (
        f"핵심 지표\n"
        f"────────────\n"
        f"정확도: {accuracy*100:.1f}% ({int(round(accuracy*len(samples)))}/{len(samples)})\n"
        f"Macro-F1: {macro_f1:.3f}\n\n"
        f"핵심 결론\n"
        f"────────────\n"
        f"촬영 각도(원근 왜곡)로 높이가\n"
        f"눌려 보여 대부분 mug로 오분류.\n"
        f"2D 폭 프로파일 방식의\n"
        f"구조적 한계로 확인됨."
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, ha="left", va="top", fontsize=13,
            bbox=dict(boxstyle="round", fc="#F2F2F2", ec="#999999"), linespacing=1.6)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
