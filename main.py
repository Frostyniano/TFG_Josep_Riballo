import os
from PIL import Image

# Ruta a la carpeta raíz de tu dataset
data_dir = "./data"

# Función para cargar las imágenes
def load_images_from_folder(folder_path):
    images = []
    labels = []
    for label in os.listdir(folder_path):  # Itera sobre las clases (0, 1, ..., 7)
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                try:
                    img = Image.open(img_path).convert("RGB")  # Abre la imagen y la convierte a RGB
                    images.append(img)
                    labels.append(int(label))  # Añade la clase como etiqueta
                except Exception as e:
                    print(f"Error al cargar la imagen {img_path}: {e}")
    return images, labels

# Carga de los conjuntos train, val y test
train_images, train_labels = load_images_from_folder(os.path.join(data_dir, "train"))
val_images, val_labels = load_images_from_folder(os.path.join(data_dir, "val"))
test_images, test_labels = load_images_from_folder(os.path.join(data_dir, "test"))

print(f"Cargadas {len(train_images)} imágenes de entrenamiento")
print(f"Cargadas {len(val_images)} imágenes de validación")
print(f"Cargadas {len(test_images)} imágenes de prueba")
