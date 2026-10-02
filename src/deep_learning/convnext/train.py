"""ConvNeXt-Tiny 전이학습 — convnext 트랙 (텀블러 형태 4종 분류).

`convnext` 브랜치(주피터 노트북 `convnext_v3.ipynb`~`v5.ipynb`, 체크포인트·원본
이미지가 그대로 커밋돼 있던 상태)에서 핵심 로직만 뽑아 dl2_resnet/efficientnet과
같은 `.py` 스크립트+이 프로젝트의 data/preprocess 기반 stratified_split(seed=42)
구조로 재작성했다. 데이터 분할 알고리즘이 다른 트랙과 동일해서 Test1(Val)은
ResNet/ViT/EfficientNet과 같은 127장이 그대로 재현된다.

2단계 학습(팀원 노트북 방식 그대로):
    Stage 1 Warmup — classifier head만 unfreeze, 짧게 학습해 새 head를 적응시킴
    Stage 2 Finetune — 전체 unfreeze + LLRD(Layer-wise LR Decay: 하위 stage일수록
        낮은 LR)로 사전학습 특징을 깨지 않으면서 전체를 미세조정

증강 스택은 팀원 v4 노트북(RandomPerspective + RandomErasing + LLRD, PPT
"v3→v4" 단계)을 그대로 가져왔다. **주의**: PPT는 마지막 단계("v4→최종")가
"val loss 기준 early stopping"이라고 설명하지만, 실제 `convnext_v5.ipynb`
코드는 그 대신 validation을 아예 없애고 train_acc 기준으로 멈추는 다른 방식을
쓰고 있어 PPT 설명과 노트북 코드가 서로 다르다 — 검증 데이터를 미리 보고 멈추는
게 아니라 완전히 held-out으로 유지하는 게 더 안전한 관행이라고 판단해, 이
스크립트는 **PPT가 설명한 대로** val loss 기준 best-model 선택 + early stopping을
구현했다(v5 노트북 코드를 그대로 베끼지 않음 — 불일치는 의도적 선택).

출력:
    checkpoints/convnext_tiny_shape.pth
    reports/figures/convnext/training_curves.png
    reports/figures/convnext/val_confusion_matrix.png
    reports/figures/convnext/test_confusion_matrix.png (--test-root 있을 때)

실행:
    python src/deep_learning/convnext/train.py
    python src/deep_learning/convnext/train.py --eval-only --test-root data/test1_orientation_backup
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CLASSES = ["straight", "taper_smooth", "taper_step", "mug"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def exif_safe_loader(path: str):
    """cv2와 달리 PIL은 기본적으로 EXIF 방향을 안 따르므로 명시적으로 보정한다."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def list_samples(root: str):
    """root/<class>/*.jpg|png 를 (path, label) 리스트로 수집."""
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
        image = exif_safe_loader(path)
        return self.transform(image), label


def stratified_split(samples, val_ratio: float, seed: int):
    """클래스별로 나눠서 섞은 뒤 val_ratio만큼 val로 뺀다 (dl2_resnet/vit/efficientnet과
    동일 알고리즘 — 같은 seed=42면 Test1(Val) 127장이 트랙 간에 완전히 동일하게 재현됨)."""
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


def build_model():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, len(CLASSES))
    return model


def build_llrd_param_groups(model, base_lr: float, decay: float):
    """ConvNeXt의 `features` 4-stage 구조에 레이어별 감쇠 LR을 적용한다 —
    앞쪽(저수준 특징) stage일수록 낮은 LR, classifier는 base_lr 그대로.
    (팀원 노트북 build_llrd_params와 동일 로직)"""
    param_groups = []
    stages = list(model.features.children())
    n = len(stages)
    for i, stage in enumerate(stages):
        lr_i = base_lr * (decay ** (n - 1 - i))
        param_groups.append({"params": list(stage.parameters()), "lr": lr_i})
    param_groups.append({"params": list(model.classifier.parameters()), "lr": base_lr})
    return param_groups


def run_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def run_eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
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


def plot_training_curves(history, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].axvline(history["warmup_epochs"], color="gray", linestyle="--", alpha=0.5, label="warmup 끝")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].axvline(history["warmup_epochs"], color="gray", linestyle="--", alpha=0.5, label="warmup 끝")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
    parser.add_argument("--test-root", default="data/test1_orientation_backup")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-full", type=float, default=5e-4, help="finetune 단계 classifier LR (base_lr) — LLRD로 하위 stage는 이보다 낮게 감쇠됨")
    parser.add_argument("--llrd-decay", type=float, default=0.8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="val loss가 이 횟수만큼 연속 개선 안 되면 조기 종료")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports/figures")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--eval-only", action="store_true",
                         help="재학습 없이 기존 체크포인트를 불러와 val/test만 재평가한다")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"model: convnext_tiny, eval_only={args.eval_only}")

    samples = list_samples(args.data_root)
    print(f"{args.data_root}: 총 {len(samples)}장")
    train_samples, val_samples = stratified_split(samples, args.val_split, args.seed)
    print(f"train {len(train_samples)}장 / val {len(val_samples)}장 (seed={args.seed})\n")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.35, p=0.4),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_loader = DataLoader(ShapeDataset(val_samples, eval_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "convnext_tiny_shape.pth")

    if args.eval_only:
        if not os.path.isfile(ckpt_path):
            print(f"체크포인트가 없습니다: {ckpt_path} (먼저 --eval-only 없이 학습해라)")
            return
        print(f"체크포인트 로드: {ckpt_path} (재학습 건너뜀)")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        _, best_val_acc = run_eval_epoch(model, val_loader, criterion, device)
        print(f"val_acc={best_val_acc:.3f}\n")
        history = None
    else:
        train_loader = DataLoader(ShapeDataset(train_samples, train_tf), batch_size=args.batch_size,
                                   shuffle=True, num_workers=0)

        tmp_ckpt_path = ckpt_path + ".tmp"
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                   "warmup_epochs": args.warmup_epochs}
        t_start = time.time()
        best_val_loss = float("inf")
        best_val_acc = 0.0

        # ── Stage 1: Warmup — classifier head만 unfreeze
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr_head, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.warmup_epochs, eta_min=1e-6)

        print("=== Stage 1: Warmup (head만 학습) ===")
        for epoch in range(1, args.warmup_epochs + 1):
            train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"[warmup] epoch {epoch:3d}/{args.warmup_epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
            if val_loss < best_val_loss:
                best_val_loss, best_val_acc = val_loss, val_acc
                torch.save(model.state_dict(), tmp_ckpt_path)

        # ── Stage 2: Finetune — 전체 unfreeze + LLRD, val loss 기준 early stopping
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            build_llrd_param_groups(model, base_lr=args.lr_full, decay=args.llrd_decay),
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-7)

        print("\n=== Stage 2: Finetune (LLRD, val loss 기준 early stopping) ===")
        no_improve = 0
        for epoch in range(1, args.finetune_epochs + 1):
            train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            elapsed = time.time() - t_start
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss, best_val_acc = val_loss, val_acc
                no_improve = 0
                torch.save(model.state_dict(), tmp_ckpt_path)
            else:
                no_improve += 1
            print(f"[finetune] epoch {epoch:3d}/{args.finetune_epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  {'(최고 갱신)' if improved else f'(개선 없음 {no_improve}/{args.patience})'}"
                  f"  ({elapsed:.0f}s 누적)")
            if no_improve >= args.patience:
                print(f"\nEarly stopping @ epoch {epoch} (val loss {args.patience}회 연속 개선 없음)")
                break

        os.replace(tmp_ckpt_path, ckpt_path)  # 전체 학습이 끝까지 성공했을 때만 최종 반영
        print(f"\n학습 완료. 최저 val_loss={best_val_loss:.4f}(val_acc={best_val_acc:.3f}), 체크포인트: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    out_dir = os.path.join(args.out_dir, "convnext")  # reports/figures/convnext/
    os.makedirs(out_dir, exist_ok=True)
    if history is not None:
        curves_path = os.path.join(out_dir, "training_curves.png")
        plot_training_curves(history, curves_path)
        print(f"학습 곡선 저장: {curves_path}")

    val_matrix = evaluate_confusion(model, val_loader, device)
    val_cm_path = os.path.join(out_dir, "val_confusion_matrix.png")
    plot_confusion(val_matrix, "ConvNeXt-Tiny Val Confusion Matrix", val_cm_path)
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
        _, test_acc = run_eval_epoch(model, test_loader, criterion, device)
        test_matrix = evaluate_confusion(model, test_loader, device)
        test_cm_path = os.path.join(out_dir, "test_confusion_matrix.png")
        plot_confusion(test_matrix, "ConvNeXt-Tiny Test(실촬영) Confusion Matrix", test_cm_path)
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
