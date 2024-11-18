import os
import pandas as pd
from collections import defaultdict

class FERPlusDataset:
    def __init__(self, data_dir, subset):
        """
        Args:
            data_dir (str): Carpeta base del dataset (ex. D:/Clase/UAB/TFG/FERPlus)
            subset (str): Subconjunt a utilitzar ('Train', 'Valid', 'Test')
        """
        self.data_dir = data_dir
        self.subset = subset
        
        # Ruta del fitxer CSV d'etiquetes
        self.labels_path = os.path.join(data_dir, f"FER2013{self.subset}", "label.csv")
        self.data = pd.read_csv(self.labels_path)  # Carrega les etiquetes des del CSV

    def count_images_per_emotion(self):
        """
        Compta el nombre d'imatges assignades a cada emoció segons la nova regla.
        """
        emotion_counts = defaultdict(int)
        emotion_labels = [
            "Neutral", "Happiness", "Surprise", "Sadness",
            "Anger", "Disgust", "Fear", "Contempt", "Unknown", "Non-Face"
        ]
        
        # Recorrem cada fila del CSV per determinar l'emoció dominant segons la nova regla
        for _, row in self.data.iterrows():
            labels = row.iloc[2:].values.astype(int)  # Columnes d'emocions
            sorted_indices = labels.argsort()[-2:]  # Índexs de les dues emocions amb més vots
            most_voted = sorted_indices[-1]
            second_most_voted = sorted_indices[-2]
            
            # Comprovem si el més votat supera al segon més votat per almenys un 30% (3 vots)
            if labels[most_voted] >= labels[second_most_voted] + 3:
                dominant_emotion = emotion_labels[most_voted]
            else:
                dominant_emotion = "Unknown"
            
            emotion_counts[dominant_emotion] += 1  # Comptem aquesta emoció

        return emotion_counts

# Comptar per cada subcarpeta
if __name__ == "__main__":
    data_dir = "D:/Clase/UAB/TFG/FERPlus"
    subsets = ["Train", "Valid", "Test"]
    
    for subset in subsets:
        print(f"\nSubcarpeta: {subset}")
        dataset = FERPlusDataset(data_dir, subset)
        emotion_counts = dataset.count_images_per_emotion()
        
        # Mostrem el nombre d'imatges per emoció dominant
        for emotion, count in emotion_counts.items():
            print(f"{emotion}: {count} imatges")
