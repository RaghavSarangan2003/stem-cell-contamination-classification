import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from model import build_efficientnet_b0
from image_transforms import get_val_test_transforms
import config

# Grad-CAM Implementation

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam, class_idx


# Image Loading

def load_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = get_val_test_transforms()
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor

# Visualization

def visualize_gradcam(image, cam, class_name, save_path):
    image_np = np.array(image)

    h, w, _ = image_np.shape

    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Grad-CAM ({class_name})")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Main

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = build_efficientnet_b0(
        num_classes=3,
        freeze_backbone=True
    ).to(device)

    # Enable gradients for Grad-CAM
    for param in model.parameters():
        param.requires_grad = True

    checkpoint = torch.load("best_model_phase1.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Target layer: last convolution block
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    # Class names (must match training order)
    class_names = ["bacteria", "healthy", "microplasma"]


    # CHANGE THIS IMAGE NAME

    test_image_name = "" # <-- replace with real image
    test_image_path = config.raw_images_path / test_image_name

    image, tensor = load_image(test_image_path, device)

    cam, class_idx = gradcam.generate(tensor)
    class_name = class_names[class_idx]

    visualize_gradcam(
        image=image,
        cam=cam,
        class_name=class_name,
        save_path=f"gradcam_{class_name}.png"
    )
