import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model import build_efficientnet_b0
from dataloader import get_dataloaders
import config


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Train / Validate Functions
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ---------------------------
# Plotting
# ---------------------------
def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase-1: Training vs Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Phase-1: Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------
# Main (Windows-safe)
# ---------------------------
if __name__ == "__main__":
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model (Phase-1: backbone frozen)
    model = build_efficientnet_b0(
        num_classes=3,
        freeze_backbone=True
    ).to(device)

    # Loss & Optimizer (classifier only)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=16,      # increase to 32 if GPU allows
        num_workers=2       # safe on Windows (guarded by __main__)
    )

    NUM_EPOCHS = config.NUM_EPOCHS

    # Metric storage
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                },
                "best_model_phase1.pth"
            )
            print("Best model saved")

        torch.cuda.empty_cache()

    # ---------------------------
    # Save metrics + plots
    # ---------------------------
    metrics_df = pd.DataFrame({
        "epoch": range(1, NUM_EPOCHS + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
    })
    metrics_df.to_csv("training_metrics_phase1.csv", index=False)

    plot_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        out_path="../assets/training/training_curves_phase1.png"
    )

    print("\nPhase-1 training completed.")
    print("Best Validation Accuracy:", best_val_acc)
