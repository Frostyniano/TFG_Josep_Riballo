import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import wandb

# Dataset FERPlus amb suport per a probabilitats
class FERPlusDataset(Dataset):
    def __init__(self, data_dir, subset, transform=None):
        self.data_dir = data_dir
        self.subset = subset
        self.transform = transform
        self.labels_path = os.path.join(data_dir, f"FER2013{self.subset}", "label.csv")
        self.data = pd.read_csv(self.labels_path, header=None)  # Llegeix sense capçalera
        self.emotion_labels = [
            "Neutral", "Happiness", "Surprise", "Sadness",
            "Anger", "Disgust", "Fear", "Contempt", "Unknown", "Non-Face"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, f"FER2013{self.subset}", self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        labels = self.data.iloc[idx, 2:].values.astype(int)

        total_votes = labels.sum()
        if total_votes > 0:
            probabilities = torch.tensor(labels / total_votes, dtype=torch.float32)
        else:
            probabilities = torch.zeros(len(labels), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return img_name, image, probabilities

# Model ResNet adaptat per a prediccions de probabilitats
class FERPlusResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(FERPlusResNet, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Funció d'entrenament per multilabel amb probabilitats
def train(config=None):
    wandb.init(project="ferplus-sweep-logloss", config=config)
    config = wandb.config

    data_dir = "D:/Clase/UAB/TFG/FERPlus"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FERPlusDataset(data_dir, "Train", transform=transform)
    valid_dataset = FERPlusDataset(data_dir, "Valid", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = FERPlusResNet(num_classes=10, dropout_rate=config.dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_mult)

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for _, images, labels in tqdm(train_loader, desc=f"Entrenament Època {epoch + 1}/{config.num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, images, labels in tqdm(valid_loader, desc="Validació"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(valid_loader)

        wandb.log({
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Learning Rate": optimizer.param_groups[0]['lr'],
            "Epoch": epoch + 1
        })

        print(f"Època {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'Validation Loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {'values': [32, 64]},
            'lr': {'values': [0.001, 0.0005]},
            'num_epochs': {'value': 10},
            'dropout_rate': {'values': [0.3, 0.5]},
            'T_0': {'values': [5, 10]},
            'T_mult': {'values': [1, 2]},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="ferplus-sweep-logloss")
    wandb.agent(sweep_id, train)
