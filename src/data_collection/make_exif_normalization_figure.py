"""EXIF 방향 정규화 단계(데이터 전처리 2단계) 발표자료용 요약 그림을 만든다.

① data/test1 158장 중 EXIF Orientation 태그가 있던 비율(파이 차트)
② 정규화 전/후 ResNet-18 Test 정확도 비교(검증 결과 — 결과에 영향 없었음을 보여줌)

실행:
    python src/data_collection/make_exif_normalization_figure.py
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 이번 세션에서 실측한 값 — data/test1(158장) EXIF 태그 분포와 정규화 전/후 검증 결과.
TOTAL = 158
ROTATED = 135  # orientation=6 (회전 필요했던 사진)
NORMAL = TOTAL - ROTATED

VERIFICATION = [
    ("정규화 전\n(원본, 135장)", 0.822, 0.825),
    ("정규화 후\n(158장 전체)", 0.823, 0.818),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="reports/figures/data_pipeline/exif_normalization_summary.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("데이터 전처리 2단계 — EXIF 방향 정규화", fontsize=16, fontweight="bold")

    # ① 파이 차트
    ax = axes[0]
    sizes = [ROTATED, NORMAL]
    labels = [f"회전 메타데이터 보유\n{ROTATED}장 ({ROTATED/TOTAL*100:.0f}%)",
              f"정상\n{NORMAL}장 ({NORMAL/TOTAL*100:.0f}%)"]
    colors = ["#E8A33D", "#4C72B0"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1.5), textprops={"fontsize": 11})
    ax.set_title(f"data/test1 {TOTAL}장 중 EXIF Orientation 분포", fontsize=13)

    # ② 검증 결과 (막대그래프 + 텍스트)
    ax = axes[1]
    x = range(len(VERIFICATION))
    acc = [v[1] * 100 for v in VERIFICATION]
    macro_f1 = [v[2] for v in VERIFICATION]
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], acc, width, label="정확도(%)", color="#4C72B0")
    ax2 = ax.twinx()
    bars2 = ax2.bar([i + width / 2 for i in x], macro_f1, width, label="Macro-F1", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels([v[0] for v in VERIFICATION], fontsize=10)
    ax.set_ylabel("정확도(%)")
    ax2.set_ylabel("Macro-F1")
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 1.0)
    ax.set_title("ResNet-18 재검증: 정규화 전/후\n(오차범위 내 동일 → 결과에 영향 없음 확인)", fontsize=12)
    for b, v in zip(bars1, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
    for b, v in zip(bars2, macro_f1):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
