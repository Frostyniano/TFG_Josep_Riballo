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
import csv

# Dataset FERPlus
class FERPlusDataset(Dataset):
    def __init__(self, data_dir, subset, transform=None):
        self.data_dir = data_dir
        self.subset = subset
        self.transform = transform
        self.labels_path = os.path.join(data_dir, f"FER2013{self.subset}", "label.csv")
        self.data = pd.read_csv(self.labels_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, f"FER2013{self.subset}", self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        labels = self.data.iloc[idx, 2:].values.astype(float)
        if self.transform:
            image = self.transform(image)
        return image, labels

# Model amb Dropout i 10 classes
class FERPlusResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):  # Incloem el paràmetre per al dropout
        super(FERPlusResNet, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Afegim una capa de Dropout abans de la capa final
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Funció d'entrenament amb guardat de resultats per època
def train_model(data_dir, batch_size=32, lr=0.001, num_epochs=10, dropout_rate=0.5, csv_file="results.csv", config=None):
    """
    Entrena el model FERPlusResNet amb el dataset FER+ i guarda els resultats per època al CSV.
    """
    # Transformacions per a les imatges
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionem les imatges
        transforms.RandomHorizontalFlip(),  # Augmentació amb flip horitzontal
        transforms.RandomRotation(15),  # Rotació aleatòria
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Ajust de lluminositat i contrast
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalització estàndard
    ])

    # Carreguem el dataset
    train_dataset = FERPlusDataset(data_dir, "Train", transform=transform)
    valid_dataset = FERPlusDataset(data_dir, "Valid", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Inicialització del model
    model = FERPlusResNet(num_classes=10, dropout_rate=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Configuració de pèrdua i optimitzador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Guardar separador i configuració al CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["-" * 50])
        writer.writerow(["Batch Size", "Learning Rate", "Epochs", "Dropout Rate"])
        writer.writerow([config["batch_size"], config["lr"], config["num_epochs"], config["dropout_rate"]])
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

    # Entrenament i guardat per època
    for epoch in range(num_epochs):
        # Entrenament
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Entrenament Època {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.argmax(dim=1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validació
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Validació"):
                images, labels = images.to(device), labels.argmax(dim=1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(valid_loader)

        # Guardar resultats per època al CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss, val_accuracy])

        print(f"Època {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return model

# Exemple d'execució
if __name__ == "__main__":
    data_dir = "D:/Clase/UAB/TFG/FERPlus"
    results_file = "results.csv"

    # Configuracions a provar
    configs = [
        {"batch_size": 64, "lr": 0.001, "num_epochs": 50, "dropout_rate": 0.3},
        {"batch_size": 64, "lr": 0.001, "num_epochs": 50, "dropout_rate": 0.5},
        {"batch_size": 64, "lr": 0.001, "num_epochs": 50, "dropout_rate": 0.2},
        {"batch_size": 64, "lr": 0.0005, "num_epochs": 50, "dropout_rate": 0.3},
        {"batch_size": 64, "lr": 0.0005, "num_epochs": 50, "dropout_rate": 0.5},
        {"batch_size": 64, "lr": 0.0005, "num_epochs": 50, "dropout_rate": 0.2},
    ]

    # Entrenar cada configuració
    for config in configs:
        print(f"\nProva amb configuració: {config}")
        train_model(
            data_dir=data_dir,
            batch_size=config["batch_size"],
            lr=config["lr"],
            num_epochs=config["num_epochs"],
            dropout_rate=config["dropout_rate"],
            csv_file=results_file,
            config=config
        )
        print(f"Configuració completada: {config}")
