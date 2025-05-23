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

class NABirdsDataset(Dataset):
    def __init__(self, data_root, split_flag, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.split_flag = split_flag  # "1" for train, "0" for test
        
        # Load images.txt -> image_id to relative image path (including subfolder)
        images_txt = self.data_root / "images.txt"
        self.image_id_to_path = {}
        with open(images_txt, 'r') as f:
            for line in f:
                img_id, img_rel_path = line.strip().split()
                self.image_id_to_path[img_id] = img_rel_path

        # Load train_test_split.txt -> image_id to train/test split
        split_txt = self.data_root / "train_test_split.txt"
        self.image_id_to_split = {}
        with open(split_txt, 'r') as f:
            for line in f:
                img_id, split_val = line.strip().split()
                self.image_id_to_split[img_id] = split_val

        # Load image_class_labels.txt -> image_id to class_id
        labels_txt = self.data_root / "image_class_labels.txt"
        self.image_id_to_class = {}
        with open(labels_txt, 'r') as f:
            for line in f:
                img_id, class_id = line.strip().split()
                self.image_id_to_class[img_id] = int(class_id)

        # Load classes.txt -> list of classes and create class_id to zero-based index mapping
        classes_txt = self.data_root / "classes.txt"
        self.class_id_to_idx = {}
        self.idx_to_class_id = []
        with open(classes_txt, 'r') as f:
            for idx, line in enumerate(f):
                class_id, class_name = line.strip().split(' ', 1)
                class_id = int(class_id)
                self.class_id_to_idx[class_id] = idx
                self.idx_to_class_id.append(class_id)

        # Filter images based on split_flag
        self.image_ids = [
            img_id for img_id, split_val in self.image_id_to_split.items() if split_val == self.split_flag
        ]

        # Sanity check: keep only images for which we have all data and that file exists
        filtered = []
        for img_id in self.image_ids:
            if img_id in self.image_id_to_path and img_id in self.image_id_to_class:
                img_path = self.data_root / "images" / self.image_id_to_path[img_id]
                if img_path.exists():
                    filtered.append(img_id)
        self.image_ids = filtered

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_rel_path = self.image_id_to_path[img_id]
        img_path = self.data_root / "images" / img_rel_path
        image = Image.open(img_path).convert("RGB")
        label_class_id = self.image_id_to_class[img_id]
        label = self.class_id_to_idx[label_class_id]

        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def train(model, dataloader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss/(loop.n+1))

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loop.set_postfix(accuracy=correct / total)
    print(f"Test Accuracy: {correct / total:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap='Blues', norm='log', cbar=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate NABirds classifier")
    parser.add_argument('-t', '--train', type=str, help='Path to save trained model')
    parser.add_argument('-e', '--evaluate', type=str, help='Path to load model for evaluation')
    parser.add_argument('--data-root', type=str, default="data/nabirds", help='Root folder of NABirds dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)  # Increased default epochs
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        transform = get_transforms(train=True)
        train_dataset = NABirdsDataset(args.data_root, split_flag="1", transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        num_classes = len(train_dataset.idx_to_class_id)
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train(model, train_loader, criterion, optimizer, device, epochs=args.epochs)
        torch.save(model.state_dict(), args.train)
        print(f"Model saved to {args.train}")

    elif args.evaluate:
        transform = get_transforms(train=False)
        test_dataset = NABirdsDataset(args.data_root, split_flag="0", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        num_classes = len(test_dataset.idx_to_class_id)
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        model.load_state_dict(torch.load(args.evaluate, map_location=device))
        evaluate(model, test_loader, device)

    else:
        print("Please specify --train or --evaluate mode.")

if __name__ == "__main__":
    main()