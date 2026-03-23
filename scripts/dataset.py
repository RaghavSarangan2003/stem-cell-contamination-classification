import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StemCellDataset(Dataset):
    """
    PyTorch Dataset for Stem Cell Image Classification
    """

    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (str): Path to split CSV (train/val/test)
            image_dir (str or Path): Directory containing images
            transform (callable, optional): Image transforms
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        # First column must be filename
        self.image_names = self.df["filename"].values

        # Convert one-hot labels to class index
        label_cols = ["bacteria", "healthy", "microplasma"]
        self.labels = self.df[label_cols].values.argmax(axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")

        # Get label
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label



