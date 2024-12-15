import cv2
import torch
import numpy as np
from torchvision import transforms
from model_ferplus_multilabel import FERPlusResNet

# Inicialitza el model entrenat
model_path = "D:/Clase/UAB/TFG/model_entrenat.pth"  # Path del model entrenat
try:
    model = FERPlusResNet(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print(f"Error: El fitxer del model no s'ha trobat a '{model_path}'. Comprova el camí especificat.")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Transformacions per pre-processar els fotogrames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_frame(face):
    """
    Processa una cara detectada per passar-la pel model.
    """
    input_tensor = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return output.cpu().numpy()

# Inicialitza el detector de cares
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Captura de vídeo en temps real
cap = cv2.VideoCapture(0)  # Usa la càmera per defecte

if not cap.isOpened():
    print("Error: No es pot accedir a la càmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No es poden obtenir fotogrames.")
        break

    # Converteix a escala de grisos per detectar cares
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=7, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Dibuixa un rectangle al voltant de la cara
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extreu la regió de la cara
        face = frame[y:y + h, x:x + w]

        # Processa la cara detectada
        results = process_frame(face)

        # Troba l'emoció amb la probabilitat més alta
        emotions = ["Neutral", "Happiness", "Surprise", "Sadness",
                    "Anger", "Disgust", "Fear", "Contempt", "Unknown", "Non-Face"]
        emotion_idx = np.argmax(results)
        emotion = emotions[emotion_idx]

        # Mostra l'emoció detectada sobre el rectangle
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Mostra el fotograma processat
    cv2.imshow('Video en Temps Real', frame)

    # Sortida si l'usuari prem 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
