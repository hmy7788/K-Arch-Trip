"""룰베이스 파이프라인을 테스트1(같은 도메인)과 테스트2(직접 촬영)에서 각각
무작위로 한 장씩 뽑아 나란히 비교하는 그림을 만든다.

각 행: 원본 사진 -> 전경 마스크 -> 폭 프로파일(판정 근거). 같은 파이프라인이
도메인에 따라 마스크·폭 프로파일이 어떻게 달라지는지 한눈에 보여준다.

실행:
    python src/rule_based/compare_test1_test2_example.py
    python src/rule_based/compare_test1_test2_example.py --seed 7   # 다른 조합으로 재추첨
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

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(__file__))
from shape_classifier import classify_shape, get_mask, SHAPE_LABELS_KO  # noqa: E402
from evaluate_test import imread_unicode, list_samples, stratified_val_split  # noqa: E402


def pick_and_process(path: str, true_cls: str):
    image_bgr = imread_unicode(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = get_mask(image_bgr)
    result = classify_shape(mask)
    return image_rgb, mask, result


def draw_row(fig_axes, image_rgb, mask, result, true_cls, row_title, filename):
    ax_img, ax_mask, ax_profile = fig_axes

    ax_img.imshow(image_rgb)
    ax_img.set_title(f"{row_title}\n원본: {filename}", fontsize=11)
    ax_img.axis("off")

    ax_mask.imshow(mask, cmap="gray")
    ax_mask.set_title("전경 마스크", fontsize=11)
    ax_mask.axis("off")

    profile = result["width_profile"]
    ax_profile.barh(range(len(profile)), profile, color="#4C72B0")
    ax_profile.invert_yaxis()
    ax_profile.set_yticks(range(len(profile)))
    ax_profile.set_yticklabels([f"구간{i}" for i in range(len(profile))], fontsize=8)
    ax_profile.set_xlabel("폭(px)")
    pred = result["shape"]
    mark = "O" if pred == true_cls else "X"
    color = "green" if pred == true_cls else "red"
    ax_profile.set_title("폭 프로파일 -> 판정", fontsize=11)
    info = (f"실제: {SHAPE_LABELS_KO[true_cls]}({true_cls})\n"
            f"예측: {SHAPE_LABELS_KO[pred]}({pred}) {mark}\n"
            f"h/d = {result['height_diameter_ratio']:.2f}\n"
            f"아래/위 폭비 = {result['bottom_top_ratio']:.2f}")
    ax_profile.text(0.97, 0.03, info, transform=ax_profile.transAxes, ha="right", va="bottom",
                     fontsize=9, color=color,
                     bbox=dict(boxstyle="round", fc="#FFF6D9", ec=color))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/preprocess", help="테스트1(같은 도메인) 소스")
    parser.add_argument("--test-root", default="data/test1_orientation_backup", help="테스트2(직접 촬영) 소스")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42, help="dl2_resnet/train.py와 동일해야 같은 127장이 재현됨")
    parser.add_argument("--seed", type=int, default=None, help="예시를 뽑는 무작위 시드(생략 시 매번 다르게 뽑힘)")
    parser.add_argument("--out", default="reports/figures/rule_based/test1_vs_test2_example.png")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    test1_samples = stratified_val_split(list_samples(args.data_root), args.val_split, args.split_seed)
    test2_samples = list_samples(args.test_root)

    path1, cls1 = rng.choice(test1_samples)
    path2, cls2 = rng.choice(test2_samples)

    print(f"테스트1 예시: [{cls1}] {path1}")
    img1, mask1, result1 = pick_and_process(path1, cls1)

    print(f"테스트2 예시: [{cls2}] {path2}")
    img2, mask2, result2 = pick_and_process(path2, cls2)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    fig.suptitle("룰베이스 파이프라인 — 테스트1(같은 도메인) vs 테스트2(직접 촬영) 무작위 예시", fontsize=14, fontweight="bold")

    draw_row(axes[0], img1, mask1, result1, cls1, "테스트1 (같은 도메인)", os.path.basename(path1))
    draw_row(axes[1], img2, mask2, result2, cls2, "테스트2 (직접 촬영)", os.path.basename(path2))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
