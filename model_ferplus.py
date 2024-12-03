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

# Dataset FERPlus amb suport per a diferents modes d'entrenament
class FERPlusDataset(Dataset):
    def __init__(self, data_dir, subset, transform=None, mode="default"):
        """
        Inicialització del dataset amb diferents modes d'entrenament:
        - "default": Aplica la regla actual d'emoció dominant amb 3 o més vots de diferència.
        - "threshold": Inclou només imatges amb una emoció que té un 70% o més dels vots.
        - "all": Utilitza totes les imatges, independentment de la distribució de vots.
        """
        self.data_dir = data_dir
        self.subset = subset
        self.transform = transform
        self.labels_path = os.path.join(data_dir, f"FER2013{self.subset}", "label.csv")
        self.data = pd.read_csv(self.labels_path)
        self.emotion_labels = [
            "Neutral", "Happiness", "Surprise", "Sadness",
            "Anger", "Disgust", "Fear", "Contempt", "Unknown", "Non-Face"
        ]
        self.mode = mode

        if self.mode == "default":
            self.filtered_data = self.data[self.data.apply(self._filter_unknown, axis=1)].reset_index(drop=True)
        elif self.mode == "threshold":
            self.filtered_data = self.data[self.data.apply(self._filter_threshold, axis=1)].reset_index(drop=True)
        elif self.mode == "all":
            self.filtered_data = self.data.reset_index(drop=True)
        else:
            raise ValueError(f"Mode no reconegut: {self.mode}")

    def _filter_unknown(self, row):
        labels = row.iloc[2:].values.astype(int)
        sorted_indices = labels.argsort()[-2:]  # Índexs de les dues emocions més votades
        most_voted = sorted_indices[-1]
        second_most_voted = sorted_indices[-2]

        if labels[most_voted] >= labels[second_most_voted] + 3:
            emotion_idx = most_voted
        else:
            emotion_idx = self.emotion_labels.index("Unknown")

        return emotion_idx != self.emotion_labels.index("Unknown")

    def _filter_threshold(self, row):
        labels = row.iloc[2:].values.astype(int)
        total_votes = labels.sum()
        if total_votes == 0:
            return False
        max_votes = labels.max()
        return (max_votes / total_votes) >= 0.7  # Comprovar si una emoció té un 70% o més dels vots

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, f"FER2013{self.subset}", self.filtered_data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        labels = self.filtered_data.iloc[idx, 2:].values.astype(int)

        if self.mode in ["default", "threshold", "all"]:
            sorted_indices = labels.argsort()[-2:]
            most_voted = sorted_indices[-1]
            second_most_voted = sorted_indices[-2]

            if labels[most_voted] >= labels[second_most_voted] + 3 or self.mode == "threshold":
                emotion_idx = most_voted
            else:
                emotion_idx = self.emotion_labels.index("Unknown")

            one_hot_label = torch.zeros(len(self.emotion_labels))
            one_hot_label[emotion_idx] = 1

        if self.transform:
            image = self.transform(image)
        return image, one_hot_label

# Model amb Dropout i 10 classes
class FERPlusResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(FERPlusResNet, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Funció d'entrenament amb guardat de resultats per època
def train_model(data_dir, batch_size=32, lr=0.001, num_epochs=10, dropout_rate=0.5, csv_file="results.csv", config=None, mode="default"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FERPlusDataset(data_dir, "Train", transform=transform, mode=mode)
    valid_dataset = FERPlusDataset(data_dir, "Valid", transform=transform, mode=mode)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = FERPlusResNet(num_classes=10, dropout_rate=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("No se detectó GPU.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["-" * 50])
        writer.writerow(["Batch Size", "Learning Rate", "Epochs", "Dropout Rate"])
        writer.writerow([config["batch_size"], config["lr"], config["num_epochs"], config["dropout_rate"]])
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

    for epoch in range(num_epochs):
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

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss, val_accuracy])

        print(f"Època {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return model

# Exemple d'ús
if __name__ == "__main__":
    data_dir = "D:/Clase/UAB/TFG/FERPlus"
    modes = ["threshold", "all","default"]
    configs = [
        {"batch_size": 64, "lr": 0.0001, "num_epochs": 50, "dropout_rate": 0.3}
    ]

    for mode in modes:
        print(f"\nEntrenant amb mode: {mode}")
        for config in configs:
            train_model(
                data_dir=data_dir,
                batch_size=config["batch_size"],
                lr=config["lr"],
                num_epochs=config["num_epochs"],
                dropout_rate=config["dropout_rate"],
                csv_file=f"results_{mode}.csv",
                config=config,
                mode=mode
            )
