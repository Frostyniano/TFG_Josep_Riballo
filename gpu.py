import torch

print(torch.__version__)
if torch.cuda.is_available():
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
else:
    print("No se detectó GPU.")
