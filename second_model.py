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

        # Load image paths
        images_txt = self.data_root / "images.txt"
        self.image_id_to_path = {}
        with open(images_txt, 'r') as f:
            for line in f:
                img_id, img_rel_path = line.strip().split()
                self.image_id_to_path[img_id] = img_rel_path

        # Load split info
        split_txt = self.data_root / "train_test_split.txt"
        self.image_id_to_split = {}
        with open(split_txt, 'r') as f:
            for line in f:
                img_id, split_val = line.strip().split()
                self.image_id_to_split[img_id] = split_val

        # Load class labels
        labels_txt = self.data_root / "image_class_labels.txt"
        self.image_id_to_class = {}
        with open(labels_txt, 'r') as f:
            for line in f:
                img_id, class_id = line.strip().split()
                self.image_id_to_class[img_id] = int(class_id)

        # Load class index mapping
        classes_txt = self.data_root / "classes.txt"
        self.class_id_to_idx = {}
        self.idx_to_class_id = []
        with open(classes_txt, 'r') as f:
            for idx, line in enumerate(f):
                class_id, class_name = line.strip().split(' ', 1)
                class_id = int(class_id)
                self.class_id_to_idx[class_id] = idx
                self.idx_to_class_id.append(class_id)

        # Load bounding boxes
        self.bboxes = {}
        bboxes_txt = self.data_root / "bounding_boxes.txt"
        with open(bboxes_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    img_id, x, y, w, h = parts
                    self.bboxes[img_id] = (int(x), int(y), int(w), int(h))

        # Filter valid images
        self.image_ids = [
            img_id for img_id, split_val in self.image_id_to_split.items() if split_val == self.split_flag
        ]
        self.image_ids = [
            img_id for img_id in self.image_ids
            if img_id in self.image_id_to_path and img_id in self.image_id_to_class
            and (self.data_root / "images" / self.image_id_to_path[img_id]).exists()
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_rel_path = self.image_id_to_path[img_id]
        img_path = self.data_root / "images" / img_rel_path
        image = Image.open(img_path).convert("RGB")

        # Crop using bounding box if available
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

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap='Blues', norm='log', cbar=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate NABirds classifier with DenseNet201")
    parser.add_argument('-t', '--train', type=str, help='Path to save/load trained model checkpoint')
    parser.add_argument('-e', '--evaluate', type=str, help='Path to load model for evaluation')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--data-root', type=str, default="data/nabirds", help='Root folder of NABirds dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        transform = get_transforms(train=True)
        train_dataset = NABirdsDataset(args.data_root, split_flag="1", transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        num_classes = len(train_dataset.idx_to_class_id)
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        start_epoch = 0
        if args.resume and os.path.exists(args.train):
            print(f"Resuming training from {args.train}")
            checkpoint = torch.load(args.train, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs + start_epoch):
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs + start_epoch}")
            model.train()
            running_loss = 0.0
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
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        num_classes = len(test_dataset.idx_to_class_id)
        model = models.densenet201(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model = model.to(device)

        checkpoint = torch.load(args.evaluate, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate(model, test_loader, device)

    else:
        print("Please specify --train or --evaluate mode.")

if __name__ == "__main__":
    main()