## Created by Claude

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def heatmap(self, input_tensor, class_idx=None):
        """Return a HxW heatmap in [0,1] for one image (input_tensor: 1x3xHxW)."""
        self.model.eval()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # weight each feature map by its mean gradient, then ReLU the combination
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def overlay_heatmap(pil_img, cam, alpha=0.45):
    """Blend a [0,1] heatmap over a PIL image; returns an RGB uint8 array."""
    import matplotlib.cm as cm
    img = np.asarray(pil_img.convert("RGB").resize((cam.shape[1], cam.shape[0]))) / 255.0
    heat = cm.jet(cam)[:, :, :3]  # drop alpha channel
    blended = (1 - alpha) * img + alpha * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
