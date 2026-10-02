"""EXIF 방향 정규화의 실제 회전 전/후 사진을 나란히 보여준다.

data/test1_orientation_backup(원본, EXIF Orientation=6 태그만 있고 픽셀은
안 돌아간 raw 상태)와 data/test1(정규화 후, 픽셀 자체가 회전 반영됨)에서
같은 파일을 각각 읽어 비교한다. "전"은 EXIF 태그를 무시하고 raw 그대로
읽어서(옆으로 누운 원래 센서 방향), "후"는 정규화된 파일을 그대로 읽는다.

실행:
    python src/data_collection/make_exif_before_after_figure.py
    python src/data_collection/make_exif_before_after_figure.py --rel-path taper_step/KakaoTalk_20260915_114226226.jpg
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rel-path", default="straight/KakaoTalk_20260915_114237129_04.jpg",
                         help="data/test1_orientation_backup, data/test1 공통 상대 경로")
    parser.add_argument("--before-root", default="data/test1_orientation_backup")
    parser.add_argument("--after-root", default="data/test1")
    parser.add_argument("--out", default="reports/figures/data_pipeline/exif_before_after_example.png")
    args = parser.parse_args()

    before_path = os.path.join(args.before_root, args.rel_path)
    after_path = os.path.join(args.after_root, args.rel_path)
    print(f"회전 전(raw, EXIF 태그 무시): {before_path}")
    print(f"회전 후(정규화됨): {after_path}")

    # EXIF 태그를 일부러 무시하고 raw 픽셀 그대로 읽는다 — exif_transpose()를 안 씀.
    with Image.open(before_path) as img:
        before_img = img.convert("RGB")
        before_size = before_img.size

    with Image.open(after_path) as img:
        after_img = img.convert("RGB")
        after_size = after_img.size

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    fig.suptitle("EXIF 방향 정규화 — 회전 전/후 실제 사진", fontsize=16, fontweight="bold")

    axes[0].imshow(before_img)
    axes[0].axis("off")
    axes[0].set_title(f"회전 전 (raw 픽셀, EXIF 태그 무시하고 읽음)\n크기: {before_size[0]}x{before_size[1]} (가로)",
                       fontsize=12, color="#B00020")

    axes[1].imshow(after_img)
    axes[1].axis("off")
    axes[1].set_title(f"회전 후 (정규화됨, 픽셀 자체가 회전 반영)\n크기: {after_size[0]}x{after_size[1]} (세로)",
                       fontsize=12, color="#1B5E20")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
