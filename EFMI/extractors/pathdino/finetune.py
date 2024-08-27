import torch
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from functools import partial
from extractor import extract_features
import classifier.svm as svm
from utils.plotting import *
from dataset.PatchedDataset import PatchedDataset
from model.PathDino import get_pathDino_model

class PathDINOForClassification(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.backbone = pretrained_model
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels, _, _ in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

def train_test_split_loaders(full_dataset, train_ratio):
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2
    )

    return train_loader, test_loader

def main(root_dir):
    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-2
    num_classes = 2
    num_images=24

    # Paths
    weights_path = "./extractors/pathdino/model/PathDino512.pth"
    
    pathdino, dino_transform = get_pathDino_model(
        weights_path=weights_path
    )

    pathdino.cuda()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the fine-tuning model
    model = PathDINOForClassification(pathdino, num_classes).to(device)

    dataset = PatchedDataset(
        root_dir=root_dir, num_images=num_images, transform=dino_transform
    )

    train_loader, test_loader = train_test_split_loaders(dataset, 0.8)

    # Set up optimizer, scheduler, and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step()

    print("Fine-tuning completed!")

    # Save the fine-tuned model
    torch.save(model.state_dict(), f"./finetuned_pathdino_{num_epochs}.pth")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PathDino")
  parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to directory containing the patches folders",
  )
  args = parser.parse_args()

  main(args.root_dir)
