"""직접 설계한 SimpleCNN 학습 — custom_cnn 트랙 (텀블러 형태 4종 분류).

구현 공부/실험 목적. 4트랙 공식 비교(dl1_maskrcnn/dl2_resnet/efficientnet)에는
포함하지 않는다 — 사전학습 없이 data/preprocess만으로 처음부터 학습해서,
ResNet 전이학습 대비 얼마나 차이 나는지 보는 대조군 실험이다(model.py 참고).

data/preprocess(강한 신호 제외된 라벨)로 train/val을 클래스별 층화 분할하고,
data/test(실제 촬영)로 domain shift까지 평가한다. dl2_resnet과 같은 이유로
학습 증강에 RandomPerspective를 넣었다(촬영 각도 왜곡이 이 프로젝트의 제일
큰 실패 원인으로 확인됨, docs/troubleshooting.md).

출력:
    checkpoints/custom_cnn_shape.pth
    reports/figures/custom_cnn/training_curves.png
    reports/figures/custom_cnn/val_confusion_matrix.png
    reports/figures/custom_cnn/test_confusion_matrix.png (data/test 있을 때)

실행:
    python src/deep_learning/custom_cnn/train.py --epochs 40
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
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from model import SimpleCNN  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CLASSES = ["straight", "taper_smooth", "taper_step", "mug"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 128  # 사전학습 없이 처음부터 학습하므로 224 대신 더 작은 입력(데이터가 적을수록 유리)


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
    """클래스별로 나눠서 섞은 뒤 val_ratio만큼 val로 뺀다 (개체 단위 분할은 아님 —
    preprocess에 물건 식별 정보가 없어서 이미지 단위 분할로 근사, 알려진 한계)."""
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


def plot_training_curves(history, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
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
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--epochs", type=int, default=40)  # 사전학습 없이 처음부터 학습이라 ResNet(20)보다 더 필요
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports/figures")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--eval-only", action="store_true",
                         help="재학습 없이 기존 체크포인트를 불러와 val/test만 재평가한다")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"model: custom_cnn (SimpleCNN, 사전학습 없음), eval_only={args.eval_only}")

    samples = list_samples(args.data_root)
    print(f"{args.data_root}: 총 {len(samples)}장")
    train_samples, val_samples = stratified_split(samples, args.val_split, args.seed)
    print(f"train {len(train_samples)}장 / val {len(val_samples)}장 (seed={args.seed})\n")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.35, p=0.4),  # 촬영 각도 왜곡을 겨냥한 증강
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 256 / 224)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_loader = DataLoader(ShapeDataset(val_samples, eval_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "custom_cnn_shape.pth")

    if args.eval_only:
        if not os.path.isfile(ckpt_path):
            print(f"체크포인트가 없습니다: {ckpt_path} (먼저 --eval-only 없이 학습해라)")
            return
        print(f"체크포인트 로드: {ckpt_path} (재학습 건너뜀)")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        _, best_val_acc = run_epoch(model, val_loader, criterion, None, device, train=False)
        print(f"val_acc={best_val_acc:.3f}\n")
        history = None
    else:
        train_loader = DataLoader(ShapeDataset(train_samples, train_tf), batch_size=args.batch_size,
                                   shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # 학습 중에는 임시 파일에만 저장한다 — 도중에 죽어도 기존 체크포인트가
        # 안 망가지도록 함 (dl2_resnet에서 실제로 겪은 사고, docs/troubleshooting.md)
        tmp_ckpt_path = ckpt_path + ".tmp"
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        t_start = time.time()
        best_val_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            elapsed = time.time() - t_start
            print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({elapsed:.0f}s 누적)")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), tmp_ckpt_path)

        os.replace(tmp_ckpt_path, ckpt_path)  # 전체 학습이 끝까지 성공했을 때만 최종 반영
        print(f"\n학습 완료. 최고 val_acc={best_val_acc:.3f}, 체크포인트: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    out_dir = os.path.join(args.out_dir, "custom_cnn")
    os.makedirs(out_dir, exist_ok=True)
    if history is not None:
        curves_path = os.path.join(out_dir, "training_curves.png")
        plot_training_curves(history, curves_path)
        print(f"학습 곡선 저장: {curves_path}")

    val_matrix = evaluate_confusion(model, val_loader, device)
    val_cm_path = os.path.join(out_dir, "val_confusion_matrix.png")
    plot_confusion(val_matrix, "custom_cnn Val Confusion Matrix", val_cm_path)
    print(f"Val confusion matrix 저장: {val_cm_path}")

    print("\nVal 클래스별 precision/recall/F1")
    for cls, (p, r, f1) in precision_recall_f1(val_matrix).items():
        print(f"  {cls:14s} precision={p:.3f} recall={r:.3f} f1={f1:.3f}")

    test_samples = list_samples(args.test_root)
    if test_samples:
        print(f"\n{args.test_root}: 총 {len(test_samples)}장 - domain shift 평가")
        test_loader = DataLoader(ShapeDataset(test_samples, eval_tf), batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        _, test_acc = run_epoch(model, test_loader, criterion, None, device, train=False)
        test_matrix = evaluate_confusion(model, test_loader, device)
        test_cm_path = os.path.join(out_dir, "test_confusion_matrix.png")
        plot_confusion(test_matrix, "custom_cnn Test(실촬영) Confusion Matrix", test_cm_path)
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
