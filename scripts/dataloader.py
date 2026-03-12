from torch.utils.data import DataLoader
from dataset import StemCellDataset
from image_transforms import (
    get_train_transforms,
    get_val_test_transforms
)
import config

def get_dataloaders(batch_size=16, num_workers=2):
    train_dataset = StemCellDataset(
        csv_path=config.train_clean_csv_path,
        image_dir=config.raw_images_path,
        transform=get_train_transforms()
    )

    val_dataset = StemCellDataset(
        csv_path=config.val_clean_csv_path,
        image_dir=config.raw_images_path,
        transform=get_val_test_transforms()
    )

    test_dataset = StemCellDataset(
        csv_path=config.test_clean_csv_path,
        image_dir=config.raw_images_path,
        transform=get_val_test_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch labels:", labels)
