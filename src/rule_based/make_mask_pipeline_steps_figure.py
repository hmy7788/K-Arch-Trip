"""마스크 생성 파이프라인(Otsu / Canny / GrabCut) 각 단계를 사진으로 보여준다.

get_mask() 내부의 세 단계를 하나씩 뜯어서 원본 옆에 나란히 배치한다 —
Otsu와 Canny는 서로 독립적인 1차 추정치이고, GrabCut이 그 둘을 합친
결과를 시드로 받아 최종 정제한다는 흐름을 한 장으로 보여준다.

실행:
    python src/rule_based/make_mask_pipeline_steps_figure.py
    python src/rule_based/make_mask_pipeline_steps_figure.py --image data/preprocess/mug/108.jpg
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
from shape_classifier import (  # noqa: E402
    _rough_mask_otsu, _rough_mask_canny, _grabcut_refine, get_mask, MAX_PROCESSING_DIM,
)

STEPS_DESC = {
    "otsu": "Otsu\n명도 기준 자동 이진화\n(밝기 대비 큰 배경에 강함)",
    "canny": "Canny\n엣지(경계선) 검출\n(그림자·저대비 배경에 강함)",
    "grabcut": "GrabCut\nOtsu+Canny를 시드로\n색 분포 기반 최종 정제",
}


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="data/preprocess/mug/108.jpg")
    parser.add_argument("--out", default="reports/figures/rule_based/mask_pipeline_steps.png")
    args = parser.parse_args()

    print(f"처리 중: {args.image}")
    image_bgr = imread_unicode(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # get_mask()와 동일하게 축소(있다면) — 내부 함수들을 직접 호출하려면 축소를 맞춰줘야 함.
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    proc = image_bgr
    if longest > MAX_PROCESSING_DIM:
        scale = MAX_PROCESSING_DIM / longest
        proc = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    otsu_mask = _rough_mask_otsu(proc)
    canny_mask = _rough_mask_canny(proc)
    rough = cv2.bitwise_or(otsu_mask, canny_mask)
    grabcut_mask = _grabcut_refine(proc, rough)
    final_mask = get_mask(image_bgr)  # 실제 파이프라인 최종 결과(컨투어 정리까지 포함)

    panels = [
        ("원본", image_rgb, False),
        (STEPS_DESC["otsu"], otsu_mask, True),
        (STEPS_DESC["canny"], canny_mask, True),
        (STEPS_DESC["grabcut"] + "\n(최종 마스크)", final_mask, True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("마스크 생성 파이프라인 — Otsu / Canny / GrabCut 단계별 결과", fontsize=16, fontweight="bold")

    for ax, (title, img, is_mask) in zip(axes, panels):
        if is_mask:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
