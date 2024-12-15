import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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
    wandb.init(project="ferplus-ResNet50", config=config)
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

    y_true, y_pred = [], []

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
        epoch_y_true, epoch_y_pred = [], []
        with torch.no_grad():
            for _, images, labels in tqdm(valid_loader, desc="Validació"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Per avaluació
                epoch_y_true.extend(labels.cpu().numpy().argmax(axis=1))
                epoch_y_pred.extend(outputs.cpu().numpy().argmax(axis=1))

        y_true.extend(epoch_y_true)
        y_pred.extend(epoch_y_pred)

        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(valid_loader)

        # Calcular mètriques
        acc = accuracy_score(epoch_y_true, epoch_y_pred)
        cm = confusion_matrix(epoch_y_true, epoch_y_pred)
        report = classification_report(epoch_y_true, epoch_y_pred, target_names=train_dataset.emotion_labels, output_dict=True, zero_division=0)
        macro_avg = report['macro avg']['f1-score'] if 'macro avg' in report else 0.0
        micro_avg = report['weighted avg']['f1-score'] if 'weighted avg' in report else 0.0

        # Registre a wandb
        wandb.log({
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Learning Rate": optimizer.param_groups[0]['lr'],
            "Accuracy": acc,
            "Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=epoch_y_true,
                preds=epoch_y_pred,
                class_names=train_dataset.emotion_labels
            ),
            "Macro Accuracy": macro_avg,
            "Micro Accuracy": micro_avg
        })

        print(f"Època {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {acc:.4f}")

    # Avaluació final
    final_acc = accuracy_score(y_true, y_pred)
    final_cm = confusion_matrix(y_true, y_pred)
    final_report = classification_report(y_true, y_pred, target_names=train_dataset.emotion_labels, output_dict=True, zero_division=0)
    wandb.log({
        "Final Accuracy": final_acc,
        "Final Confusion Matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=train_dataset.emotion_labels
        ),
        "Class Distribution": {label: sum(1 for y in y_true if y == idx) for idx, label in enumerate(train_dataset.emotion_labels)}
    })

if __name__ == "__main__":
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'Validation Loss',
            'goal': 'minimize'
        },
        'parameters': {
            'batch_size': {'values': [128]},
            'lr': {'values': [0.001, 0.0005]},
            'num_epochs': {'value': 10},
            'dropout_rate': {'values': [0.3, 0.5]},
            'T_0': {'values': [5, 10]},
            'T_mult': {'values': [1, 2]},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="ferplus-ResNet50")
    wandb.agent(sweep_id, train)
