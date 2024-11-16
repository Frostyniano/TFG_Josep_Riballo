import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FERPlusDataset(Dataset):
    def __init__(self, data_dir, subset, transform=None):
        """
        Args:
            data_dir (str): Carpeta base del dataset (ex. D:/Clase/UAB/TFG/FERPlus)
            subset (str): Subconjunt a utilitzar ('Train', 'Valid', 'Test')
            transform (callable, optional): Transformacions a aplicar a les imatges
        """
        self.data_dir = data_dir
        self.subset = subset
        self.transform = transform
        
        # Ruta del fitxer CSV d'etiquetes
        self.labels_path = os.path.join(data_dir, "data", f"FER2013{self.subset}", "label.csv")
        self.data = pd.read_csv(self.labels_path)  # Carrega les etiquetes des del CSV

    def __len__(self):
        """Retorna el nombre total d'imatges en el conjunt."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retorna una imatge i les seves etiquetes corresponents.
        """
        # Construcció del camí de la imatge
        img_name = os.path.join(
            self.data_dir,f"FER2013{self.subset}", self.data.iloc[idx, 0]
        )
        image = Image.open(img_name).convert("RGB")
        
        # Carregar les etiquetes (a partir de la tercera columna)
        labels = self.data.iloc[idx, 2:].values.astype(float)
        
        # Aplicar transformacions si existeixen
        if self.transform:
            image = self.transform(image)
        
        return image, labels

# Exemple d'ús
if __name__ == "__main__":
    # Transformacions per a les imatges
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionem a 224x224 (estàndard ResNet)
        transforms.ToTensor(),         # Convertim a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalització
    ])
    
    # Creem el dataset per al conjunt de prova
    dataset = FERPlusDataset("D:/Clase/UAB/TFG/FERPlus", "Test", transform=transform)
    
    # Mostrem la mida del conjunt i una mostra
    print(f"Nombre d'imatges al conjunt de prova: {len(dataset)}")
    
    # Primera imatge i etiquetes
    img, label = dataset[0]
    print(f"Forma de la imatge: {img.shape}")
    print(f"Etiquetes: {label}")
