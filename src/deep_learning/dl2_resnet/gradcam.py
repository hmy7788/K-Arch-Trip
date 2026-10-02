"""ResNet 판정 근거를 Grad-CAM으로 시각화한다.

`layer4`(마지막 conv 블록)의 활성화·그래디언트로 히트맵을 만들어 원본
이미지에 겹쳐 보여준다 — 모델이 이미지의 어느 부분을 보고 형태를
판정했는지 확인하는 용도. 클래스별로 몇 장씩 뽑아 원본/히트맵/예측-정답을
한 그리드에 저장한다(맞은 것과 틀린 것 섞어서 — 틀렸을 때 뭘 보고
헷갈렸는지가 더 흥미로움).

실행:
    python src/deep_learning/dl2_resnet/gradcam.py --split test2 --per-class 4   # 직접 촬영(기본값)
    python src/deep_learning/dl2_resnet/gradcam.py --split test1 --per-class 4   # 같은 도메인 Val
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
from train import (  # noqa: E402
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    exif_safe_loader,
    list_samples,
    stratified_split,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rule_based"))
from shape_classifier import SHAPE_LABELS_KO  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


class GradCAM:
    """target_layer의 활성화·그래디언트로 클래스별 히트맵을 만든다."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        pred_idx = int(output.argmax(dim=1).item())
        target_idx = pred_idx if class_idx is None else class_idx
        output[0, target_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, pred_idx, output.softmax(dim=1)[0].detach().cpu().numpy()


def overlay_cam(rgb_uint8: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam_resized = cv2.resize(cam, (rgb_uint8.shape[1], rgb_uint8.shape[0]))
    heatmap = (plt.get_cmap("jet")(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.45 * heatmap + 0.55 * rgb_uint8).astype(np.uint8)
    return overlay


def preprocess(image_rgb: np.ndarray, device: str):
    """train.py의 eval_tf와 동일한 224x224 crop을 쓰되, 정규화 전 RGB도 같이 반환한다."""
    resized = cv2.resize(image_rgb, (256, 256))
    top = (256 - 224) // 2
    cropped = resized[top:top + 224, top:top + 224]

    tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to(device)
    tensor.requires_grad_(False)  # Grad-CAM은 activation의 그래디언트만 필요, 입력 자체는 필요 없음
    return tensor, cropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["test1", "test2"], default="test2",
                         help="test1=같은 도메인 분할(data/preprocess의 Val), test2=직접 촬영(--test-root)")
    parser.add_argument("--test-root", default="data/test1", help="--split test2일 때만 사용")
    parser.add_argument("--data-root", default="data/preprocess", help="--split test1일 때만 사용 — train.py와 동일해야 같은 Val이 재현됨")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42, help="train.py의 --seed와 동일해야 같은 Val 분할이 재현됨")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--checkpoint", default=None, help="생략 시 checkpoints/<model>_shape.pth")
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="그리드에 표시할 샘플을 고르는 셔플 시드")
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args()

    ckpt_path = args.checkpoint or os.path.join("checkpoints", f"{args.model}_shape.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"체크포인트 로드: {ckpt_path}")

    model = build_model(args.model, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    target_layer = model.layer4[-1]  # 마지막 conv 블록 — 공간 정보가 남아있는 마지막 지점
    cam_engine = GradCAM(model, target_layer)

    if args.split == "test1":
        all_samples = list_samples(args.data_root)
        _, samples = stratified_split(all_samples, args.val_split, args.split_seed)
        print(f"테스트1(같은 도메인 Val): {args.data_root} 중 {len(samples)}장 (val-split={args.val_split}, seed={args.split_seed})")
    else:
        samples = list_samples(args.test_root)
        print(f"테스트2(직접 촬영): {args.test_root} 중 {len(samples)}장")
    # samples는 (path, 정수 라벨) 리스트
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
            image_rgb = cv2.cvtColor(exif_safe_loader_to_bgr(path), cv2.COLOR_BGR2RGB)
            tensor, cropped_rgb = preprocess(image_rgb, device)

            cam, pred_idx, probs = cam_engine(tensor)
            overlay = overlay_cam(cropped_rgb, cam)

            ax.imshow(overlay)
            pred_cls = CLASSES[pred_idx]
            mark = "O" if pred_cls == cls else "X"
            color = "green" if pred_cls == cls else "red"
            ax.set_title(f"실제:{SHAPE_LABELS_KO[cls]}\n예측:{SHAPE_LABELS_KO[pred_cls]}({probs[pred_idx]:.2f}) {mark}",
                          fontsize=9, color=color)
            ax.axis("off")
            print(f"  [{cls}] {os.path.basename(path)}: 예측={pred_cls} ({'맞음' if mark=='O' else '틀림'})")

    plt.tight_layout()
    out_dir = os.path.join(args.out_dir, args.model)  # reports/figures/<model>/ — 모델별로 폴더 분리
    os.makedirs(out_dir, exist_ok=True)
    out_name = "gradcam_test1_val.png" if args.split == "test1" else "gradcam_test2.png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nGrad-CAM 그리드 저장: {out_path}")


def exif_safe_loader_to_bgr(path: str) -> np.ndarray:
    """exif_safe_loader(PIL, RGB)를 cv2 스타일 BGR ndarray로 바꾼다."""
    pil_img = exif_safe_loader(path)
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    main()
