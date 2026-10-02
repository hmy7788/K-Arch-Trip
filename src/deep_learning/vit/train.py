"""DeiT-Small/16 전이학습 — vit 트랙 (텀블러 형태 4종 분류).

`taehyun/ViT` 브랜치의 로직(모델·증강·차등 학습률)을 가져오되, 데이터는 이
프로젝트의 다른 트랙들과 동일하게 data/preprocess(642장, 강한 신호 제외)를
층화 분할해서 쓴다 — taehyun 브랜치 자체 데이터(441장, 별도 수집분)와는 다름.

기본값은 "실험 B"(백본 마지막 4블록 + norm + head만 unfreeze, 차등 학습률)를
ImageNet 사전학습 가중치에서 바로 시작하는 구성 — taehyun 브랜치의
run_training_experiment(experiment="B", ...)와 동일한 하이퍼파라미터
(batch_size=8, epochs=25, lr_blocks=1e-5, lr_head=1e-4, weight_decay=0.01).

실행:
    python src/deep_learning/vit/train.py
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from model import (  # noqa: E402
    CLASSES, create_deit_model, freeze_backbone_for_linear_probe,
    unfreeze_last_blocks_for_fine_tuning, unfreeze_all, get_optimizer_param_groups,
)
from transforms import get_train_transforms, get_val_transforms, get_plain_transforms  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def exif_safe_loader(path: str):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def list_samples(root: str):
    samples = []
    for cls in CLASSES:
        class_dir = os.path.join(root, cls)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png"):
                samples.append((os.path.join(class_dir, name), CLASS_TO_IDX[cls]))
    return samples


class ShapeDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(exif_safe_loader(path)), label


def stratified_split(samples, val_ratio: float, seed: int):
    by_class = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))
    rng = random.Random(seed)
    train, val = [], []
    for items in by_class.values():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


def evaluate_confusion(model, loader, device):
    model.eval()
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            for t, p in zip(labels.numpy(), preds):
                matrix[t, p] += 1
    return matrix


def plot_confusion(matrix, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("예측"); ax.set_ylabel("실제"); ax.set_title(title)
    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if matrix[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/preprocess")
    parser.add_argument("--test-root", default="data/test1")
    parser.add_argument("--experiment", choices=["A", "B"], default="B",
                         help="A=백본 전체 동결(head만 학습), B=마지막 4블록+norm+head unfreeze(차등 LR) — --baseline이면 무시됨")
    parser.add_argument("--baseline", action="store_true",
                         help="완전 베이스라인: 종횡비 패딩/증강/부분freeze/차등LR/weight-decay 전부 제거 — "
                              "단순 Resize+CenterCrop, 증강 없음, 백본 전체를 단일 LR로 학습, weight_decay=0")
    parser.add_argument("--epochs", type=int, default=None, help="생략 시 A=10, B=25, baseline=25(taehyun 브랜치 기본값)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=None, help="생략 시 A=1e-3, B/baseline=1e-4")
    parser.add_argument("--lr-blocks", type=float, default=1e-5, help="실험 B에서 백본 블록 학습률(baseline에서는 미사용)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="baseline이면 0으로 강제됨")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports/figures")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--eval-only", action="store_true",
                         help="재학습 없이 기존 체크포인트를 불러와 Test1(Val)/Test2만 재평가한다")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    mode_label = "baseline(완전 베이스)" if args.baseline else f"experiment={args.experiment}"
    print(f"model: DeiT-Small/16, {mode_label}")

    samples = list_samples(args.data_root)
    print(f"{args.data_root}: 총 {len(samples)}장")
    train_samples, val_samples = stratified_split(samples, args.val_split, args.seed)
    print(f"train {len(train_samples)}장 / val {len(val_samples)}장 (seed={args.seed})\n")

    if args.baseline:
        # 완전 베이스라인: train/eval 둘 다 증강 없는 동일한 단순 변환(Resize+CenterCrop)
        train_tf = get_plain_transforms(224)
        eval_tf = get_plain_transforms(224)
    else:
        train_tf = get_train_transforms(224)
        eval_tf = get_val_transforms(224)

    train_loader = DataLoader(ShapeDataset(train_samples, train_tf), batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(ShapeDataset(val_samples, eval_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    model = create_deit_model(pretrained=True, num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.baseline:
        ckpt_name = "deit_small_baseline_shape.pth"
    elif args.experiment == "A":
        ckpt_name = "deit_small_expa_shape.pth"
    else:
        ckpt_name = "deit_small_expb_shape.pth"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

    if args.eval_only:
        if not os.path.isfile(ckpt_path):
            print(f"체크포인트가 없습니다: {ckpt_path} (먼저 --eval-only 없이 학습해라)")
            return
        print(f"체크포인트 로드: {ckpt_path} (재학습 건너뜀)")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        _, best_val_acc = run_epoch(model, val_loader, criterion, None, device, train=False)
        print(f"val_acc={best_val_acc:.3f}\n")
    else:
        if args.baseline:
            # 완전 베이스라인: 백본 전체를 단일 학습률로, weight_decay=0
            unfreeze_all(model)
            epochs = args.epochs or 25
            lr_head = args.lr_head or 1e-4
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_head, weight_decay=0.0)
        elif args.experiment == "A":
            freeze_backbone_for_linear_probe(model)
            epochs = args.epochs or 10
            lr_head = args.lr_head or 1e-3
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=lr_head, weight_decay=args.weight_decay)
        else:
            unfreeze_last_blocks_for_fine_tuning(model, num_blocks=4)
            epochs = args.epochs or 25
            lr_head = args.lr_head or 1e-4
            param_groups = get_optimizer_param_groups(model, lr_blocks=args.lr_blocks, lr_head=lr_head,
                                                        weight_decay=args.weight_decay)
            optimizer = torch.optim.AdamW(param_groups)

        tmp_ckpt_path = ckpt_path + ".tmp"
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        t_start = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            elapsed = time.time() - t_start
            print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({elapsed:.0f}s 누적)")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), tmp_ckpt_path)

        os.replace(tmp_ckpt_path, ckpt_path)
        print(f"\n학습 완료. 최고 val_acc={best_val_acc:.3f}, 체크포인트: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    variant = "baseline" if args.baseline else f"exp{args.experiment.lower()}"
    out_dir = os.path.join(args.out_dir, "vit", variant)
    os.makedirs(out_dir, exist_ok=True)

    if not args.eval_only:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        epochs_range = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs_range, history["train_loss"], label="train")
        axes[0].plot(epochs_range, history["val_loss"], label="val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
        axes[1].plot(epochs_range, history["train_acc"], label="train")
        axes[1].plot(epochs_range, history["val_acc"], label="val")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1); axes[1].legend()
        plt.tight_layout()
        curves_path = os.path.join(out_dir, "training_curves.png")
        fig.savefig(curves_path, dpi=120)
        plt.close(fig)
        print(f"학습 곡선 저장: {curves_path}")

    val_matrix = evaluate_confusion(model, val_loader, device)
    val_cm_path = os.path.join(out_dir, "val_confusion_matrix.png")
    plot_confusion(val_matrix, "DeiT-Small Val Confusion Matrix", val_cm_path)
    print(f"Val confusion matrix 저장: {val_cm_path}")

    print("\nVal 클래스별 precision/recall/F1")
    val_prf = precision_recall_f1(val_matrix)
    for cls, (p, r, f1) in val_prf.items():
        print(f"  {cls:14s} precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
    val_macro_f1 = sum(f1 for _, _, f1 in val_prf.values()) / len(val_prf)
    print(f"Val Macro-F1: {val_macro_f1:.3f}")

    test_samples = list_samples(args.test_root)
    if test_samples:
        print(f"\n{args.test_root}: 총 {len(test_samples)}장 - domain shift 평가")
        test_loader = DataLoader(ShapeDataset(test_samples, eval_tf), batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        _, test_acc = run_epoch(model, test_loader, criterion, None, device, train=False)
        test_matrix = evaluate_confusion(model, test_loader, device)
        test_cm_path = os.path.join(out_dir, "test_confusion_matrix.png")
        plot_confusion(test_matrix, "DeiT-Small Test(실촬영) Confusion Matrix", test_cm_path)
        print(f"Test confusion matrix 저장: {test_cm_path}")
        print("\nTest 클래스별 precision/recall/F1")
        test_prf = precision_recall_f1(test_matrix)
        for cls, (p, r, f1) in test_prf.items():
            print(f"  {cls:14s} precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
        test_macro_f1 = sum(f1 for _, _, f1 in test_prf.values()) / len(test_prf)
        print(f"Test Macro-F1: {test_macro_f1:.3f}")
        print(f"\nVal acc={best_val_acc:.3f} -> Test acc={test_acc:.3f} (하락폭 {best_val_acc - test_acc:+.3f})")
    else:
        print(f"\n{args.test_root}에 이미지가 없어 domain shift 평가를 건너뜀")


if __name__ == "__main__":
    main()
