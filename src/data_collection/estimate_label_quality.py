"""data/raw2의 폴더 라벨 품질을 자동으로 추정한다.

룰베이스(get_mask+classify_shape)와 Mask R-CNN(get_mask_maskrcnn+classify_shape)
두 방법으로 각각 판정해서 폴더 라벨과 비교한다.

- **둘 다 폴더 라벨과 다르게 판정** -> 진짜 나쁜 데이터(엉뚱한 사진 등)일 가능성이 높은 강한 신호
- **하나만 다르게 판정** -> 그 방법론의 한계일 수도 있는 약한 신호(반드시 나쁜 데이터는 아님)

사람이 직접 검수하기 전, "대충 몇 장이나 걸러질지" 감을 잡기 위한 자동 추정치일 뿐 —
진짜 정답은 아니다.

실행:
    python src/data_collection/estimate_label_quality.py --root data/raw2
"""

import argparse
import glob
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rule_based.shape_classifier import classify_shape, get_mask  # noqa: E402
from deep_learning.dl1_maskrcnn import get_mask_maskrcnn, load_model  # noqa: E402

CLASSES = ("straight", "taper_smooth", "taper_step", "mug")


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def predict(mask) -> str:
    if mask is None or not mask.any():
        return "none"
    return classify_shape(mask)["shape"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw2")
    args = parser.parse_args()

    print("Mask R-CNN 모델 로드 중...")
    model, device = load_model()
    print(f"로드 완료 (device={device})")

    files_by_class = {}
    total = 0
    for cls in CLASSES:
        files = sorted(glob.glob(os.path.join(args.root, cls, "*.jpg")))
        files_by_class[cls] = files
        total += len(files)
    print(f"전체 대상: {total}장\n")

    both_disagree = 0
    either_disagree = 0
    both_agree = 0
    per_class = {c: {"both_disagree": 0, "either": 0, "agree": 0, "total": 0} for c in CLASSES}

    processed = 0
    t_start = time.time()

    for cls in CLASSES:
        files = files_by_class[cls]
        for path in files:
            image = imread_unicode(path)
            rmask = get_mask(image)
            mmask = get_mask_maskrcnn(image, model, device)

            rpred = predict(rmask)
            mpred = predict(mmask)

            r_wrong = rpred != cls
            m_wrong = mpred != cls

            per_class[cls]["total"] += 1
            if r_wrong and m_wrong:
                both_disagree += 1
                per_class[cls]["both_disagree"] += 1
            elif r_wrong or m_wrong:
                either_disagree += 1
                per_class[cls]["either"] += 1
            else:
                both_agree += 1
                per_class[cls]["agree"] += 1

            processed += 1
            if processed % 20 == 0 or processed == total:
                elapsed = time.time() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f"  [{processed:4d}/{total}] {cls:14s} {os.path.basename(path):30s} "
                      f"룰베이스={rpred:14s} MaskRCNN={mpred:14s} "
                      f"| 경과 {elapsed:.0f}s, 남은 시간 약 {eta:.0f}s")

    print()
    print("=" * 60)
    print(f"전체 {total}장")
    print(f"  둘 다 다르게 판정(강한 신호): {both_disagree}장 ({both_disagree/total*100:.1f}%)")
    print(f"  하나만 다르게 판정(약한 신호): {either_disagree}장 ({either_disagree/total*100:.1f}%)")
    print(f"  둘 다 동의: {both_agree}장 ({both_agree/total*100:.1f}%)")
    print()
    print(f"{'클래스':14s} {'총':>5s} {'둘다다름':>10s} {'하나만':>8s} {'동의':>8s}")
    for c in CLASSES:
        d = per_class[c]
        print(f"{c:14s} {d['total']:5d} {d['both_disagree']:6d}({d['both_disagree']/d['total']*100:4.1f}%) "
              f"{d['either']:8d} {d['agree']:8d}")


if __name__ == "__main__":
    main()
