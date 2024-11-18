# TFG_Josep_Riballo
Josep Riballo TFG 2024-2025 mención de computación, detector de emociones en tiempo real con video.

Per descarregar el dataSet AffectNet https://www.kaggle.com/datasets/thienkhonghoc/affectnet?resource=download

python generate_training_data.py -d <dataset base folder> -fer <fer2013.csv path> -ferplus <fer2013new.csv path>

python D:\Clase\UAB\TFG\FERPlus\src\generate_training_data.py -d D:\Clase\UAB\TFG\FERPlus -fer D:\Clase\UAB\TFG\challenges-in-representation-learning-facial-expression-recognition-challenge\fer2013\fer2013.csv -ferplus D:\Clase\UAB\TFG\FERPlus\fer2013new.csv

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124