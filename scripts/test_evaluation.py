import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import build_efficientnet_b0
from dataloader import get_dataloaders



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # Load Model

    model = build_efficientnet_b0(
        num_classes=3,
        freeze_backbone=True
    ).to(device)

    checkpoint = torch.load("best_model_phase1.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    # Load Test Data

    _, _, test_loader = get_dataloaders(
        batch_size=16,
        num_workers=2
    )

    all_preds = []
    all_labels = []


    # Inference

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)


    # Metrics

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc:.4f}")

    class_names = ["bacteria", "healthy", "microplasma"]

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4
        )
    )


    # Confusion Matrix

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_phase1.png", dpi=300)
    plt.show()
