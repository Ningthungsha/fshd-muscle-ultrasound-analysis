import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Generates explainability heatmaps for ultrasound muscle assessment.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1].
        
        Args:
            input_tensor: [1, 1, H, W] torch tensor
            target_class: int class index or None for predicted class
            
        Returns:
            heatmap: [H, W] float numpy array in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize to input dimensions
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros((H, W), dtype=np.float32)

        return cam

    @staticmethod
    def overlay_heatmap(image_np, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
        """
        Overlays heatmap on grayscale ultrasound image.
        Ensures heatmap matches image dimensions exactly.
        """
        if image_np.ndim == 2:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image_np.copy()

        H, W = image_rgb.shape[:2]
        if heatmap.shape != (H, W):
            heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

        heatmap_u8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        heatmap_color = cv2.applyColorMap(heatmap_u8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_color, alpha, 0)
        return overlay

