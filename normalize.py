import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision

# Ruta a la carpeta raíz de tu dataset
data_dir = r"C:\Clase\TFG\data"

# Definir el Dataset personalizado
class EmotionDataset(Dataset):
    def __init__(self, folder_path, img_size=(224, 224), max_images_per_class=None):
        self.folder_path = folder_path
        self.img_size = img_size
        self.max_images_per_class = max_images_per_class
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),  # Convierte a tensor y normaliza a [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización estándar
        ])
        
        # Construimos una lista de todas las imágenes y sus etiquetas
        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if os.path.isdir(label_path):
                images = os.listdir(label_path)
                if self.max_images_per_class:
                    images = images[:self.max_images_per_class]  # Limitar el número de imágenes si se especifica
                for filename in images:
                    img_path = os.path.join(label_path, filename)
                    self.data.append((img_path, int(label)))  # Guardar ruta y etiqueta
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)  # Aplicar transformaciones
        except Exception as e:
            print(f"Error al cargar la imagen {img_path}: {e}")
            image = torch.zeros(3, *self.img_size)  # Devolver una imagen vacía en caso de error
        
        return image, label

# Función para mostrar un batch de imágenes
def show_batch(images, labels):
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_img.permute(1, 2, 0))  # Cambiar las dimensiones para visualización
    plt.title("Etiquetas: " + " ".join(str(label.item()) for label in labels))
    plt.axis("off")
    plt.show()

# Proteger el código principal
if __name__ == '__main__':
    batch_size = 64
    max_images_per_class = 100  # Cambia este valor al número máximo deseado

    train_dataset = EmotionDataset(os.path.join(data_dir, "train"), max_images_per_class=max_images_per_class)
    val_dataset = EmotionDataset(os.path.join(data_dir, "val"), max_images_per_class=max_images_per_class)
    test_dataset = EmotionDataset(os.path.join(data_dir, "test"), max_images_per_class=max_images_per_class)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Obtener y mostrar un batch de imágenes y etiquetas
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    show_batch(images, labels)
