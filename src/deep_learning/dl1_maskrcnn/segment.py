"""Mask R-CNN(ResNet50+FPN) 기반 세그멘테이션 — dl1 트랙.

COCO로 **이미 학습된 가중치를 그대로 쓴다(제로샷)** — 우리 데이터로 fine-tuning
하지 않는다. 목적은 예측된 마스크를 이진화해서
`src/rule_based/shape_classifier.py`의 `compute_width_profile`/`classify_shape`를
그대로 재사용하는 것 (CLAUDE.md에 명시된 유일한 트랙 간 코드 공유 예외).
재질 분류 헤드는 만들지 않는다 — 형태 판별에만 이 마스크를 쓴다.

텀블러는 COCO 80종에 정확히 없어서, 모양이 비슷한 카테고리(cup/bottle/vase/
wine glass/bowl) 중 점수 높은 걸 고른다. 해당 카테고리가 하나도 안 잡히면
전체 검출 중 가장 점수가 높은 걸 대신 쓴다(완전히 놓치는 것보다는 낫다는 판단).
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

CANDIDATE_CATEGORIES = {"cup", "bottle", "vase", "wine glass", "bowl"}
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

_WEIGHTS = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
CATEGORY_NAMES = _WEIGHTS.meta["categories"]


def load_model(device: str | None = None):
    """사전학습 Mask R-CNN을 로드한다. device 생략 시 CUDA 가능하면 자동 사용."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = maskrcnn_resnet50_fpn_v2(weights=_WEIGHTS)
    model.eval()
    model.to(device)
    return model, device


def get_mask_maskrcnn(
    image_bgr: np.ndarray,
    model,
    device: str,
    score_threshold: float = SCORE_THRESHOLD,
    mask_threshold: float = MASK_THRESHOLD,
) -> np.ndarray | None:
    """BGR 이미지(np.ndarray, cv2 컨벤션)를 받아 이진 마스크를 반환한다.

    검출이 하나도 없으면 None을 반환한다 (get_mask()의 "빈 마스크" 관례와
    구분하기 위함 — 아예 못 찾은 것과 빈 마스크는 다른 상황).
    """
    image_rgb = image_bgr[:, :, ::-1]
    tensor = TF.to_tensor(np.ascontiguousarray(image_rgb)).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    masks = output["masks"].cpu().numpy()  # (N, 1, H, W), soft [0, 1]

    keep = scores >= score_threshold
    if not keep.any():
        return None

    scores, labels, masks = scores[keep], labels[keep], masks[keep]
    label_names = [CATEGORY_NAMES[i] for i in labels]

    candidate_idx = [i for i, name in enumerate(label_names) if name in CANDIDATE_CATEGORIES]
    chosen_idx = max(candidate_idx, key=lambda i: scores[i]) if candidate_idx else int(np.argmax(scores))

    soft_mask = masks[chosen_idx, 0]
    binary_mask = (soft_mask >= mask_threshold).astype(np.uint8) * 255
    return binary_mask
