"""GradCAM — 모델이 사진의 어디를 보고 그렇게 판정했는지 히트맵으로 보여준다.

원본: 팀원(taein)의 streamlit_final/src/gradcam.py. 훅 거는 방식과 계산은 그대로 두었다.

원리: 마지막 합성곱 블록의 출력(활성화)과, 예측 클래스 점수를 그 활성화로 미분한 값(기울기)을
곱해 더한다. 기울기가 크다 = 그 위치가 점수를 밀어올렸다. 그래서 빨간 곳이 판정 근거다.

읽을 때 주의 — 히트맵은 '모델이 본 곳'이지 '정답의 근거'가 아니다. 배경이나 손을 보고 맞히고
있었다면 여기서 드러난다(그게 이 그림의 쓸모다). 판정이 틀렸을 때도 히트맵은 그럴듯하게 나온다.
"""
import cv2
import numpy as np
import torch


class GradCAM:
    """모델에 훅을 건다. 같은 모델에 두 번 만들면 훅이 쌓이니 하나만 만들어 재사용할 것."""

    def __init__(self, model, target_layer):
        self.model = model
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """input_tensor: (1,3,H,W) requires_grad=True → (H,W) 0~1 히트맵."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB uint8 이미지에 JET 컬러맵 히트맵을 겹친다. 반환도 RGB uint8."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)


def target_layer(model):
    """히트맵을 뽑을 마지막 합성곱 블록. ConvNeXt는 features[-1](=features[7])."""
    for attr in ("features", "layer4", "blocks"):
        block = getattr(model, attr, None)
        if block is not None:
            try:
                return block[-1]
            except (TypeError, IndexError, KeyError):
                return block
    children = list(model.children())
    return children[-2] if len(children) >= 2 else children[-1]
