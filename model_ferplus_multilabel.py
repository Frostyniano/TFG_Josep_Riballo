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

# Dataset FERPlus amb suport per a probabilitats
class FERPlusDataset(Dataset):
    def __init__(self, data_dir, subset, transform=None):
        """
        Dataset per FERPlus amb processament de vots com a probabilitats.
        """
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
        # Nom del fitxer d'imatge
        img_name = os.path.join(self.data_dir, f"FER2013{self.subset}", self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        # Vots per emoció (columna 3 en endavant)
        labels = self.data.iloc[idx, 2:].values.astype(int)

        # Normalitzar els vots a probabilitats
        total_votes = labels.sum()
        if total_votes > 0:
            probabilities = torch.tensor(labels / total_votes, dtype=torch.float32)
        else:
            probabilities = torch.zeros(len(labels), dtype=torch.float32)  # Si no hi ha vots

        # Transformació de la imatge (si escau)
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
            nn.Sigmoid()  # Sortida com a probabilitats entre 0 i 1
        )

    def forward(self, x):
        return self.model(x)

# Funció per guardar prediccions i etiquetes reals en un CSV amb salts de línia
def save_predictions_to_csv(csv_file, img_name, predictions, actual, emotion_labels):
    """
    Desa les prediccions i etiquetes reals en un fitxer CSV, separades per una línia.
    """
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

         # Escriure prediccions en una fila
        name_row = [img_name]
        writer.writerow(name_row)
        # Escriure prediccions en una fila
        pred_row = ["Prediccions"] + [f"{p:.4f}" for p in predictions]
        writer.writerow(pred_row)

        # Escriure etiquetes correctes en una fila
        actual_row = ["", "Correctes"] + [f"{a:.4f}" for a in actual]
        writer.writerow(actual_row)

        # Escriure una línia buida per separar
        writer.writerow([])

# Escriure capçalera del CSV de prediccions
def initialize_predictions_csv(results_csv, emotion_labels):
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escrivim la capçalera
        header = ["Nom Imatge"] + ["Tipus"] + emotion_labels
        writer.writerow(header)

# Funció d'entrenament per multilabel amb probabilitats
def train_model(data_dir, batch_size, lr, num_epochs, dropout_rate, csv_file, config_name, results_csv):
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = FERPlusResNet(num_classes=10, dropout_rate=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()  # Binary Cross Entropy per multilabel
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Escriure els detalls de la configuració al fitxer CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["-" * 50])
        writer.writerow(["Configuració:", config_name])
        writer.writerow(["Batch Size", "Learning Rate", "Epochs", "Dropout Rate"])
        writer.writerow([batch_size, lr, num_epochs, dropout_rate])
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for img_names, images, labels in tqdm(train_loader, desc=f"Entrenament Època {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img_names, images, labels in tqdm(valid_loader, desc="Validació"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Guardar les probabilitats predites i reals
                for i in range(len(images)):
                    predicted_probs = outputs[i].cpu().numpy()
                    actual_probs = labels[i].cpu().numpy()
                    save_predictions_to_csv(results_csv, img_names[i], predicted_probs, actual_probs, train_dataset.emotion_labels)

        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(valid_loader)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss])

        print(f"Època {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model

if __name__ == "__main__":
    data_dir = "D:/Clase/UAB/TFG/FERPlus"
    csv_file = "results_multilabel.csv"
    results_csv = "prediccions.csv"

    # Inicialitzar el CSV de prediccions
    emotion_labels = [
        "Neutral", "Happiness", "Surprise", "Sadness",
        "Anger", "Disgust", "Fear", "Contempt", "Unknown", "Non-Face"
    ]
    initialize_predictions_csv(results_csv, emotion_labels)

    # Llista de configuracions
    configs = [
        {"name": "Config 1", "batch_size": 32, "lr": 0.001, "num_epochs": 10, "dropout_rate": 0.3},
    ]

    for config in configs:
        print(f"\nEntrenant amb configuració: {config['name']}")
        train_model(
            data_dir=data_dir,
            batch_size=config["batch_size"],
            lr=config["lr"],
            num_epochs=config["num_epochs"],
            dropout_rate=config["dropout_rate"],
            csv_file=csv_file,
            config_name=config["name"],
            results_csv=results_csv
        )
