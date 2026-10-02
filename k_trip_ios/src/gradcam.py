"""Adapted from team GradCAM: temporary tensor hook avoids inference hooks."""
import cv2
import numpy as np
import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.layer = model, target_layer

    def generate(self, input_tensor, class_idx):
        captured = {}
        def save(module, args, output):
            captured["activation"] = output
        hook = self.layer.register_forward_hook(save)
        try:
            with torch.enable_grad():
                output = self.model(input_tensor)
                activation = captured["activation"]
                gradient, = torch.autograd.grad(output[0, class_idx], activation)
                cam = torch.relu((gradient.mean((2, 3), keepdim=True) * activation).sum(1))[0]
                cam = cam.detach().cpu().numpy()
                cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
                return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        finally:
            hook.remove()

def overlay_cam(image_rgb, cam, alpha=.45):
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1-alpha, heat, alpha, 0)

def target_layer(model):
    return model.features[-1]
