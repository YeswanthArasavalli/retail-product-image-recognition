import os
from pathlib import Path
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


cudnn.enabled = False
cudnn.benchmark = False


DATA_DIR = Path("data_cls")
MODEL_DIR = Path("models_torch")
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16         
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4


def get_dataloaders() -> Tuple[DataLoader, DataLoader, list]:
    # ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes: int) -> nn.Module:
    # EfficientNet-B0 pretrained on ImageNet
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"Num classes: {num_classes}")

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # Mixed precision scaler (for speed on GPU)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0

    # Save class names
    with open(MODEL_DIR / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_acc)

        print(
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                MODEL_DIR / "best_model_efficientnet_b0.pth",
            )
            print(f"New best val_acc: {best_val_acc:.4f} (model saved)")

    print("\nTraining finished. Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
