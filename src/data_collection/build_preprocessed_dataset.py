"""data/raw2에서 "강한 신호"(룰베이스+Mask R-CNN 둘 다 폴더 라벨과 다르게 판정)로
걸러진 사진만 제외하고 data/preprocess/에 복사한다.

세 가지 신호로 분류한다:
- **강한 신호**: 룰베이스·Mask R-CNN 둘 다 폴더 라벨과 다르게 판정 -> 제외
- **약한 신호**: 둘 중 하나만 다르게 판정 -> 그 방법론 하나의 한계일 수 있어서
  (특히 룰베이스가 taper_smooth/taper_step에서 약함) 제외하지 않고 포함
- **문제 없음**: 둘 다 폴더 라벨과 동일하게 판정

("강한 신호"만 나쁜 데이터로 취급하는 근거는 estimate_label_quality.py 실측,
docs/experiment-log.md 참고.)

파일별 판정 결과는 <dst>/label_quality_report.md에 표로 남긴다.
재실행하면 매번 data/preprocess/를 비우고 새로 만든다(raw2가 바뀌면 다시
돌리면 됨).

실행:
    python src/data_collection/build_preprocessed_dataset.py --src data/raw2 --dst data/preprocess
"""

import argparse
import glob
import os
import shutil
import sys
import time

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rule_based.shape_classifier import classify_shape, get_mask  # noqa: E402
from deep_learning.dl1_maskrcnn import get_mask_maskrcnn, load_model  # noqa: E402

CLASSES = ("straight", "taper_smooth", "taper_step", "mug")

STRONG = "강한 신호"
WEAK = "약한 신호"
OK = "문제 없음"


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def predict(mask) -> str:
    if mask is None or not mask.any():
        return "none"
    return classify_shape(mask)["shape"]


def signal_for(cls: str, rpred: str, mpred: str) -> str:
    r_wrong = rpred != cls
    m_wrong = mpred != cls
    if r_wrong and m_wrong:
        return STRONG
    if r_wrong or m_wrong:
        return WEAK
    return OK


def write_report(report_path: str, rows: list[dict], summary_lines: list[str]) -> None:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 라벨 품질 리포트 (data/raw2)\n\n")
        f.write("`build_preprocessed_dataset.py` 실행 결과 — 파일마다 룰베이스/Mask R-CNN 판정과 신호를 기록한다.\n\n")
        f.write("- **강한 신호**: 둘 다 폴더 라벨과 다르게 판정 → `data/preprocess/`에서 제외됨\n")
        f.write("- **약한 신호**: 하나만 다르게 판정 → 포함됨 (방법론 한계일 수 있어 자동 제외 안 함)\n")
        f.write("- **문제 없음**: 둘 다 폴더 라벨과 동일하게 판정\n\n")
        f.write("## 요약\n\n")
        for line in summary_lines:
            f.write(f"- {line}\n")
        f.write("\n## 클래스별 상세\n\n")
        for cls in CLASSES:
            f.write(f"### {cls}\n\n")
            f.write("| 파일 | 룰베이스 판정 | Mask R-CNN 판정 | 신호 |\n")
            f.write("|---|---|---|---|\n")
            for row in rows:
                if row["cls"] != cls:
                    continue
                f.write(f"| {row['file']} | {row['rpred']} | {row['mpred']} | {row['signal']} |\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/raw2")
    parser.add_argument("--dst", default="data/preprocess")
    args = parser.parse_args()

    print("Mask R-CNN 모델 로드 중...")
    model, device = load_model()
    print(f"로드 완료 (device={device})")

    if os.path.isdir(args.dst):
        print(f"{args.dst} 기존 내용 삭제 후 새로 생성")
        shutil.rmtree(args.dst)

    files_by_class = {}
    total = 0
    for cls in CLASSES:
        files = sorted(glob.glob(os.path.join(args.src, cls, "*.jpg")))
        files_by_class[cls] = files
        total += len(files)
        os.makedirs(os.path.join(args.dst, cls), exist_ok=True)
    print(f"전체 대상: {total}장\n")

    counts = {c: {STRONG: 0, WEAK: 0, OK: 0} for c in CLASSES}
    rows = []

    processed = 0
    t_start = time.time()

    for cls in CLASSES:
        for path in files_by_class[cls]:
            image = imread_unicode(path)
            rpred = predict(get_mask(image))
            mpred = predict(get_mask_maskrcnn(image, model, device))
            signal = signal_for(cls, rpred, mpred)

            counts[cls][signal] += 1
            rows.append({"cls": cls, "file": os.path.basename(path), "rpred": rpred, "mpred": mpred, "signal": signal})

            if signal != STRONG:
                dst_path = os.path.join(args.dst, cls, os.path.basename(path))
                shutil.copy2(path, dst_path)

            processed += 1
            if processed % 20 == 0 or processed == total:
                elapsed = time.time() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f"  [{processed:4d}/{total}] {cls:14s} {os.path.basename(path):30s} "
                      f"{signal:8s} | 경과 {elapsed:.0f}s, 남은 시간 약 {eta:.0f}s")

    total_strong = sum(counts[c][STRONG] for c in CLASSES)
    total_weak = sum(counts[c][WEAK] for c in CLASSES)
    total_ok = sum(counts[c][OK] for c in CLASSES)
    total_kept = total_weak + total_ok

    print()
    print("=" * 60)
    print(f"{'클래스':14s} {'강한신호':>10s} {'약한신호':>10s} {'문제없음':>10s} {'채택':>6s}")
    for c in CLASSES:
        kept = counts[c][WEAK] + counts[c][OK]
        print(f"{c:14s} {counts[c][STRONG]:10d} {counts[c][WEAK]:10d} {counts[c][OK]:10d} {kept:6d}")
    print("-" * 60)
    print(f"{'합계':14s} {total_strong:10d} {total_weak:10d} {total_ok:10d} {total_kept:6d}")
    print(f"\n결과: {args.dst}/<class>/ 에 {total_kept}장 저장됨 (강한 신호 {total_strong}장 제외)")

    summary_lines = [
        f"전체 {total}장 중 강한 신호(제외) {total_strong}장, 약한 신호(포함) {total_weak}장, 문제 없음 {total_ok}장",
        f"최종 채택: {total_kept}장 → `{args.dst}/<class>/`",
    ]
    report_path = os.path.join(args.dst, "label_quality_report.md")
    write_report(report_path, rows, summary_lines)
    print(f"리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
