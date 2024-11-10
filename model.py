import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import models
from tqdm import tqdm
import normalize

# Definir un modelo (por ejemplo, ResNet18 preentrenado para reconocimiento de emociones)
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionRecognitionModel, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Usar ResNet18 preentrenado
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Cambiar la capa final para clasificación en 8 clases

    def forward(self, x):
        return self.model(x)

# Proteger el código principal
if __name__ == '__main__':
    # Hiperparámetros y configuración
    num_epochs = 6
    learning_rate = 0.001
    batch_size = 64
    num_classes = 8  # Número de clases en tu dataset (emociones)

    # Inicializar el modelo, la función de pérdida y el optimizador
    model = EmotionRecognitionModel(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()  # Pérdida para clasificación
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Mover el modelo a la GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Inicializar DataLoaders
    max_images_per_class = 2500  # Cambia este valor al número máximo deseado

    train_dataset = normalize.EmotionDataset(os.path.join(normalize.data_dir, "train"), max_images_per_class=max_images_per_class)
    val_dataset = normalize.EmotionDataset(os.path.join(normalize.data_dir, "val"), max_images_per_class=max_images_per_class)
    train_loader = normalize.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = normalize.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Entrenamiento del modelo
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()  # Modo de entrenamiento

        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)  # Mover datos a GPU

            # Paso de entrenamiento
            optimizer.zero_grad()  # Limpiar gradientes previos
            outputs = model(images)  # Paso hacia adelante
            loss = criterion(outputs, labels)  # Calcular la pérdida
            loss.backward()  # Paso hacia atrás (gradientes)
            optimizer.step()  # Actualizar parámetros

            running_loss += loss.item() * images.size(0)  # Acumular pérdida

        # Calcular la pérdida media de entrenamiento
        train_loss = running_loss / len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluación en el conjunto de validación
        model.eval()  # Modo de evaluación
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Desactivar cálculo de gradientes en evaluación
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # Calcular la pérdida media de validación y la precisión
        val_loss /= len(val_loader.dataset)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
