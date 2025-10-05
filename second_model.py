import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

"""
This implementation uses MobileNetV2 instead of ResNet101 or DenseNet201.
The purpose of this comparison is to analyze trade-offs between:
- Model complexity vs inference/training speed
- GPU-dependent high-accuracy models vs lightweight CPU-compatible models

By running both models under similar conditions, we evaluate:
- Speed on CPU vs GPU
- Accuracy and generalization across bird species

This MobileNet version is optimized for accessibility and speed, especially on hardware-limited environments.

Test Accuracy: o.435
"""

class NABirdsDataset(Dataset):
    def __init__(self, data_root, split_flag, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.split_flag = split_flag  # "1" for train, "0" for test

        with open(self.data_root / "images.txt", 'r') as f:
            self.image_id_to_path = {img_id: path for img_id, path in (line.strip().split() for line in f)}

        with open(self.data_root / "train_test_split.txt", 'r') as f:
            self.image_id_to_split = {img_id: split for img_id, split in (line.strip().split() for line in f)}

        with open(self.data_root / "image_class_labels.txt", 'r') as f:
            self.image_id_to_class = {img_id: int(cls) for img_id, cls in (line.strip().split() for line in f)}

        self.class_id_to_idx = {}
        self.idx_to_class_id = []
        with open(self.data_root / "classes.txt", 'r') as f:
            for idx, line in enumerate(f):
                class_id, _ = line.strip().split(' ', 1)
                class_id = int(class_id)
                self.class_id_to_idx[class_id] = idx
                self.idx_to_class_id.append(class_id)

        self.bboxes = {}
        with open(self.data_root / "bounding_boxes.txt", 'r') as f:
            for line in f:
                img_id, x, y, w, h = line.strip().split()
                self.bboxes[img_id] = (int(x), int(y), int(w), int(h))

        self.image_ids = [
            img_id for img_id, split in self.image_id_to_split.items()
            if split == self.split_flag and (self.data_root / "images" / self.image_id_to_path[img_id]).exists()
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.data_root / "images" / self.image_id_to_path[img_id]
        image = Image.open(img_path).convert("RGB")

        if img_id in self.bboxes:
            x, y, w, h = self.bboxes[img_id]
            image = image.crop((x, y, x + w, y + h))

        if self.transform:
            image = self.transform(image)

        class_id = self.image_id_to_class[img_id]
        label = self.class_id_to_idx[class_id]
        return image, label

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap='Blues', norm='log', cbar=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate NABirds classifier using MobileNetV2")
    parser.add_argument('-t', '--train', type=str, help='Path to save/load trained model checkpoint')
    parser.add_argument('-e', '--evaluate', type=str, help='Path to load model for evaluation')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--data-root', type=str, default="data/nabirds", help='Root folder of NABirds dataset')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cpu")

    if args.train:
        transform = get_transforms(train=True)
        train_dataset = NABirdsDataset(args.data_root, split_flag="1", transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        num_classes = len(train_dataset.idx_to_class_id)
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        start_epoch = 0
        if args.resume and os.path.exists(args.train):
            checkpoint = torch.load(args.train, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from {args.train}, starting at epoch {start_epoch}")

        model = model.to(device)
        for epoch in range(start_epoch, args.epochs + start_epoch):
            model.train()
            running_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs + start_epoch}")
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                loop.set_postfix(loss=running_loss / (loop.n + 1))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, args.train)
        print(f"Model checkpoint saved to {args.train}")

    elif args.evaluate:
        transform = get_transforms(train=False)
        test_dataset = NABirdsDataset(args.data_root, split_flag="0", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        num_classes = len(test_dataset.idx_to_class_id)
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model = model.to(device)

        checkpoint = torch.load(args.evaluate, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        evaluate(model, test_loader)

    else:
        print("Please specify --train or --evaluate mode.")

if __name__ == "__main__":
    main()