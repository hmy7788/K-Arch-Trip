"""ViT(DeiT-Small) 판정 근거를 Grad-CAM으로 시각화한다.

`taehyun/ViT` 브랜치(`src/explainability/vit_gradcam.py`)의 patch-token 기반
Grad-CAM 로직을 이식 — ResNet과 달리 ViT는 공간 정보가 2D feature map이 아니라
197개 토큰(1 CLS + 14x14 patch)으로 흩어져 있어서, 마지막 Transformer 블록의
patch 토큰 활성화·그래디언트를 14x14로 되접어 히트맵을 만든다
(dl2_resnet/gradcam.py와 CLI 패턴은 동일하게 맞춤).

실행:
    python src/deep_learning/vit/gradcam.py --split test2 --per-class 4   # 직접 촬영(기본값)
    python src/deep_learning/vit/gradcam.py --split test1 --per-class 4   # 같은 도메인 Val
"""

import argparse
import os
import random
import sys

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")  # 리다이렉트 시 즉시 출력 + cp949 인코딩 에러 방지

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import CLASSES, create_deit_model  # noqa: E402
from train import exif_safe_loader, list_samples, stratified_split  # noqa: E402
from transforms import IMAGENET_MEAN, IMAGENET_STD, AspectRatioPadResize  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rule_based"))
from shape_classifier import SHAPE_LABELS_KO  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


class ViTGradCAM:
    """DeiT-Small의 마지막 Transformer 블록(`blocks[-1].norm1`) patch 토큰으로 히트맵을 만든다."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.blocks[-1].norm1
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        target_idx = int(torch.argmax(probs)) if class_idx is None else class_idx
        logits[0, target_idx].backward(retain_graph=True)

        # 토큰 0은 [CLS]라서 제외하고 나머지 196개 patch(14x14)만 쓴다
        act = self.activations[:, 1:, :].detach()
        grad = self.gradients[:, 1:, :].detach()
        weights = torch.mean(grad, dim=1, keepdim=True)
        cam = torch.relu(torch.sum(weights * act, dim=-1))[0].cpu().numpy()

        cam_2d = cam.reshape(14, 14)
        cam_min, cam_max = cam_2d.min(), cam_2d.max()
        cam_2d = (cam_2d - cam_min) / (cam_max - cam_min) if cam_max - cam_min > 1e-8 else np.zeros_like(cam_2d)
        cam_resized = cv2.resize(cam_2d, (224, 224), interpolation=cv2.INTER_CUBIC)
        return np.clip(cam_resized, 0.0, 1.0), target_idx, probs.detach().cpu().numpy()


def overlay_cam(rgb_uint8: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = (plt.get_cmap("jet")(cam)[:, :, :3] * 255).astype(np.uint8)
    return (0.45 * heatmap + 0.55 * rgb_uint8).astype(np.uint8)


def preprocess(image_rgb: np.ndarray, device: str):
    """train.py의 eval 변환(AspectRatioPadResize)과 동일하게 맞추되, 정규화 전 RGB도 같이 반환한다."""
    from PIL import Image
    pad_resize = AspectRatioPadResize(224)
    padded = np.array(pad_resize(Image.fromarray(image_rgb)))
    arr = padded.astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor, padded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["test1", "test2"], default="test2",
                         help="test1=같은 도메인 분할(data/preprocess의 Val), test2=직접 촬영(--test-root)")
    parser.add_argument("--test-root", default="data/test1_orientation_backup", help="--split test2일 때만 사용")
    parser.add_argument("--data-root", default="data/preprocess", help="--split test1일 때만 사용 — train.py와 동일해야 같은 Val이 재현됨")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42, help="train.py의 --seed와 동일해야 같은 Val 분할이 재현됨")
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "deit_small_expb_shape.pth"))
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="그리드에 표시할 샘플을 고르는 셔플 시드")
    parser.add_argument("--out-dir", default="reports/figures/vit")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"체크포인트 로드: {args.checkpoint}")

    model = create_deit_model(num_classes=len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    cam_engine = ViTGradCAM(model)

    if args.split == "test1":
        all_samples = list_samples(args.data_root)
        _, samples = stratified_split(all_samples, args.val_split, args.split_seed)
        print(f"테스트1(같은 도메인 Val): {args.data_root} 중 {len(samples)}장 (val-split={args.val_split}, seed={args.split_seed})")
    else:
        samples = list_samples(args.test_root)
        print(f"테스트2(직접 촬영): {args.test_root} 중 {len(samples)}장")

    by_class = {c: [p for p, label_idx in samples if CLASSES[label_idx] == c] for c in CLASSES}
    random.seed(args.seed)
    for c in CLASSES:
        random.shuffle(by_class[c])

    fig, axes = plt.subplots(len(CLASSES), args.per_class, figsize=(3 * args.per_class, 3 * len(CLASSES)))

    print(f"\nGrad-CAM 생성 중 (클래스당 {args.per_class}장)...")
    for row, cls in enumerate(CLASSES):
        paths = by_class[cls][:args.per_class]
        for col in range(args.per_class):
            ax = axes[row, col]
            if col >= len(paths):
                ax.axis("off")
                continue
            path = paths[col]
            image_rgb = np.array(exif_safe_loader(path))
            tensor, padded_rgb = preprocess(image_rgb, device)

            cam, pred_idx, probs = cam_engine(tensor)
            overlay = overlay_cam(padded_rgb, cam)

            ax.imshow(overlay)
            pred_cls = CLASSES[pred_idx]
            mark = "O" if pred_cls == cls else "X"
            color = "green" if pred_cls == cls else "red"
            ax.set_title(f"실제:{SHAPE_LABELS_KO[cls]}\n예측:{SHAPE_LABELS_KO[pred_cls]}({probs[pred_idx]:.2f}) {mark}",
                          fontsize=9, color=color)
            ax.axis("off")
            print(f"  [{cls}] {os.path.basename(path)}: 예측={pred_cls} ({'맞음' if mark == 'O' else '틀림'})")

    plt.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    out_name = "gradcam_test1_val.png" if args.split == "test1" else "gradcam_test2.png"
    out_path = os.path.join(args.out_dir, out_name)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nGrad-CAM 그리드 저장: {out_path}")


if __name__ == "__main__":
    main()
