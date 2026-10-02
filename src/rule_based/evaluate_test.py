"""룰베이스(get_mask + classify_shape)를 평가한다.

다른 트랙(dl1_maskrcnn/evaluate_test.py, dl2_resnet/train.py)과 같은 방식으로
confusion matrix를 reports/figures/rule_based/에 저장한다.

--source test1은 dl2_resnet/train.py와 정확히 동일한 stratified_split(같은
CLASSES 순서·같은 정렬·같은 seed=42)을 재현해서, data/preprocess에서 ResNet의
Val과 "완전히 똑같은 127장"을 뽑는다 — 트랙 간 코드 공유는 안 하지만(코드 분리
원칙) 알고리즘만 동일하게 복제해서 같은 파일 집합이 나오도록 함. --source test2는
기존처럼 --test-root(직접 촬영)를 그대로 평가.

실행:
    python src/rule_based/evaluate_test.py --source test2 --test-root data/test1
    python src/rule_based/evaluate_test.py --source test1
"""

import argparse
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 라벨이 깨지지 않도록
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


def stratified_val_split(samples, val_ratio: float, seed: int):
    """dl2_resnet/train.py의 stratified_split()과 알고리즘을 동일하게 복제한 것.

    라벨이 문자열(cls)이라는 점만 다르고, 클래스별 그룹화 순서(CLASSES 순서대로
    처음 등장한 순)·그룹 내 정렬·rng.shuffle 호출 순서가 전부 같아서 같은
    seed에서는 같은 파일 집합이 val로 뽑힌다. 반환값 중 val 세트만 쓴다
    (train 세트는 룰베이스가 애초에 학습을 안 하므로 불필요).
    """
    by_class = defaultdict(list)
    for path, cls in samples:
        by_class[cls].append((path, cls))
    rng = random.Random(seed)
    val = []
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
    return val


def precision_recall_f1(matrix):
    results = {}
    for i, cls in enumerate(CLASSES):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[cls] = (precision, recall, f1)
    return results


def plot_confusion(matrix, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title(title)
    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if matrix[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["test1", "test2"], default="test2",
                         help="test1=ResNet Val과 동일한 127장(data/preprocess), test2=직접 촬영(--test-root)")
    parser.add_argument("--test-root", default="data/test1", help="--source test2일 때만 사용")
    parser.add_argument("--data-root", default="data/preprocess", help="--source test1일 때만 사용")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42, help="dl2_resnet/train.py의 --seed와 동일해야 같은 127장이 재현됨")
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args()

    if args.source == "test1":
        all_samples = list_samples(args.data_root)
        samples = stratified_val_split(all_samples, args.val_split, args.split_seed)
        source_label = f"{args.data_root}(Val {len(samples)}장, ResNet 테스트1과 동일)"
    else:
        samples = list_samples(args.test_root)
        source_label = args.test_root
    print(f"{source_label}: 총 {len(samples)}장\n")

    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    correct = 0
    t_start = time.time()

    for i, (path, cls) in enumerate(samples, 1):
        image = imread_unicode(path)
        mask = get_mask(image)
        pred = classify_shape(mask)["shape"] if mask.any() else "none"

        if pred in cls_to_idx:
            matrix[cls_to_idx[cls], cls_to_idx[pred]] += 1
        correct += int(pred == cls)

        if i % 1 == 0 or i == len(samples):  # 장당 처리 시간이 길어서(GrabCut) 매 장 출력
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(samples) - i) / rate if rate > 0 else 0
            print(f"  [{i:3d}/{len(samples)}] {cls:14s} {os.path.basename(path):30s} "
                  f"예측={pred:14s} | 경과 {elapsed:.0f}s, 남은 시간 약 {eta:.0f}s")

    total = len(samples)
    print()
    print("=" * 50)
    print(f"전체 정확도: {correct}/{total} ({correct/total*100:.1f}%)")
    print()
    print("클래스별 판정 분포")
    for cls in CLASSES:
        row = matrix[cls_to_idx[cls]]
        cls_total = row.sum()
        cls_correct = row[cls_to_idx[cls]]
        breakdown = ", ".join(f"{SHAPE_LABELS_KO[CLASSES[j]]}({CLASSES[j]})={row[j]}"
                               for j in range(len(CLASSES)) if row[j] > 0)
        acc = cls_correct / cls_total * 100 if cls_total else 0
        print(f"  {cls:14s} {cls_correct}/{cls_total} ({acc:4.1f}%) — {breakdown}")

    prf = precision_recall_f1(matrix)
    print("\n클래스별 precision/recall/F1")
    for cls, (p, r, f1) in prf.items():
        print(f"  {cls:14s} precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
    macro_f1 = sum(f1 for _, _, f1 in prf.values()) / len(prf)
    print(f"\nMacro-F1: {macro_f1:.3f}")

    out_dir = os.path.join(args.out_dir, "rule_based")
    os.makedirs(out_dir, exist_ok=True)
    if args.source == "test1":
        cm_path = os.path.join(out_dir, "test1_val_confusion_matrix.png")
        plot_confusion(matrix, "룰베이스 테스트1(같은 도메인 Val) Confusion Matrix", cm_path)
    else:
        cm_path = os.path.join(out_dir, "test_confusion_matrix.png")
        plot_confusion(matrix, "룰베이스 테스트2(실촬영) Confusion Matrix", cm_path)
    print(f"\nconfusion matrix 저장: {cm_path}")


if __name__ == "__main__":
    main()
