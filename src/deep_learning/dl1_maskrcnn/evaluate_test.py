"""제로샷 Mask R-CNN vs 룰베이스 pseudo-label로 fine-tuning한 체크포인트를
data/test(직접 촬영 실사진)로 비교 평가한다.

fine-tuning 체크포인트(checkpoints/maskrcnn_finetuned_rulebase_pseudolabels.pth)는
2026-09-15 실험에서 raw2 기준으로 제로샷보다 나빴던(85%->78%) 걸로 이미 확인됐지만
(docs/experiment-log.md), data/test(실제 촬영, 크롤링 이미지와 다른 도메인)에서는
어떻게 나오는지 직접 비교해본다.

실행:
    python src/deep_learning/dl1_maskrcnn/evaluate_test.py --test-root data/test
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rule_based.shape_classifier import classify_shape, SHAPE_LABELS_KO  # noqa: E402
from deep_learning.dl1_maskrcnn import load_model, get_mask_maskrcnn  # noqa: E402

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


def load_finetuned_model(ckpt_path: str, device: str):
    """fine-tuning 때와 동일하게 배경/텀블러 2클래스로 헤드를 교체한 뒤 가중치를 불러온다."""
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.to(device)
    return model


def get_mask_finetuned(image_bgr, model, device, score_threshold=0.5, mask_threshold=0.5):
    """파인튜닝 모델은 라벨이 0/1(배경/텀블러)뿐이라 COCO 카테고리 필터링이 필요 없다."""
    image_rgb = image_bgr[:, :, ::-1]
    tensor = TF.to_tensor(np.ascontiguousarray(image_rgb)).to(device)
    with torch.no_grad():
        output = model([tensor])[0]
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    masks = output["masks"].cpu().numpy()
    keep = (scores >= score_threshold) & (labels == 1)
    if not keep.any():
        return None
    scores, masks = scores[keep], masks[keep]
    best = int(np.argmax(scores))
    return (masks[best, 0] >= mask_threshold).astype(np.uint8) * 255


def predict(mask) -> str:
    if mask is None or not mask.any():
        return "none"
    return classify_shape(mask)["shape"]


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


def macro_f1_of(matrix) -> float:
    prf = precision_recall_f1(matrix)
    return sum(f1 for _, _, f1 in prf.values()) / len(prf)


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
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--finetuned-ckpt", default="checkpoints/maskrcnn_finetuned_rulebase_pseudolabels.pth")
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("제로샷 모델 로드 중...")
    zs_model, _ = load_model(device)
    print("파인튜닝 모델 로드 중...")
    ft_model = load_finetuned_model(args.finetuned_ckpt, device)
    print("로드 완료\n")

    samples = list_samples(args.test_root)
    print(f"{args.test_root}: 총 {len(samples)}장\n")

    zs_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    ft_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

    zs_correct, ft_correct = 0, 0
    t_start = time.time()

    for i, (path, cls) in enumerate(samples, 1):
        image = imread_unicode(path)

        zs_pred = predict(get_mask_maskrcnn(image, zs_model, device))
        ft_pred = predict(get_mask_finetuned(image, ft_model, device))

        if zs_pred in cls_to_idx:
            zs_matrix[cls_to_idx[cls], cls_to_idx[zs_pred]] += 1
        if ft_pred in cls_to_idx:
            ft_matrix[cls_to_idx[cls], cls_to_idx[ft_pred]] += 1
        zs_correct += int(zs_pred == cls)
        ft_correct += int(ft_pred == cls)

        if i % 20 == 0 or i == len(samples):
            elapsed = time.time() - t_start
            print(f"  [{i:3d}/{len(samples)}] {cls:14s} {os.path.basename(path):30s} "
                  f"제로샷={zs_pred:14s} 파인튜닝={ft_pred:14s} | 경과 {elapsed:.0f}s")

    total = len(samples)
    print()
    print("=" * 60)
    print(f"{'':14s} {'제로샷':>10s} {'파인튜닝':>10s}")
    print(f"{'전체 정확도':14s} {zs_correct}/{total} ({zs_correct/total*100:4.1f}%)   "
          f"{ft_correct}/{total} ({ft_correct/total*100:4.1f}%)")
    print(f"{'Macro-F1':14s} {macro_f1_of(zs_matrix):10.3f}   {macro_f1_of(ft_matrix):10.3f}")

    out_dir = os.path.join(args.out_dir, "maskrcnn")  # reports/figures/maskrcnn/ — 모델별로 폴더 분리
    os.makedirs(out_dir, exist_ok=True)
    zs_path = os.path.join(out_dir, "zeroshot_test_confusion_matrix.png")
    ft_path = os.path.join(out_dir, "finetuned_test_confusion_matrix.png")
    plot_confusion(zs_matrix, "Mask R-CNN 제로샷 Test Confusion Matrix", zs_path)
    plot_confusion(ft_matrix, "Mask R-CNN 파인튜닝 Test Confusion Matrix", ft_path)
    print(f"\n confusion matrix 저장: {zs_path}, {ft_path}")


if __name__ == "__main__":
    main()
