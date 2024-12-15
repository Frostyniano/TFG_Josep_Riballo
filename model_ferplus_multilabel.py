import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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

# Entrenament del model
if __name__ == "__main__":
    # Configuració
    config = {
        "data_dir": "D:/Clase/UAB/TFG/FERPlus",
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 1,
        "dropout_rate": 0.5,
        "model_save_path": "model_entrenat.pth"
    }

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FERPlusDataset(config["data_dir"], "Train", transform=transform)
    valid_dataset = FERPlusDataset(config["data_dir"], "Valid", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

    model = FERPlusResNet(num_classes=10, dropout_rate=config["dropout_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Entrenament
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        for _, images, labels in tqdm(train_loader, desc=f"Entrenament Època {epoch + 1}/{config['num_epochs']}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validació
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

        print(f"Època {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Guarda el model entrenat
    torch.save(model.state_dict(), config["model_save_path"])
    print(f"Model guardat a {config['model_save_path']}")
