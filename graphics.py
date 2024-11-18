import pandas as pd
import matplotlib.pyplot as plt


def parse_results(file_path):
    data = []
    current_config = None
    config_id = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Batch Size"):
                # Llegim la configuració actual
                current_config = {}
                config_values = next(file).strip().split(',')
                current_config["Batch Size"] = int(config_values[0])
                current_config["Learning Rate"] = float(config_values[1])
                current_config["Epochs"] = int(config_values[2])
                current_config["Dropout Rate"] = float(config_values[3])
                config_id += 1  # Identificador únic per a cada configuració
                current_config["Config ID"] = config_id
            elif line.startswith("Epoch"):
                # Llegim les dades per a aquesta configuració
                epoch_data = []
                for epoch_line in file:
                    epoch_line = epoch_line.strip()
                    if epoch_line.startswith("-"):  # Final de configuració
                        break
                    # Comprovar si la línia és vàlida abans de processar-la
                    if len(epoch_line.split(',')) == 4:  # Assegurem-nos que té 4 valors
                        epoch_values = list(map(float, epoch_line.split(',')))
                        epoch_data.append(epoch_values)
                # Afegim les dades al conjunt principal
                for row in epoch_data:
                    data.append({
                        "Config ID": current_config["Config ID"],
                        "Batch Size": current_config["Batch Size"],
                        "Learning Rate": current_config["Learning Rate"],
                        "Epochs": int(row[0]),
                        "Dropout Rate": current_config["Dropout Rate"],
                        "Train Loss": row[1],
                        "Val Loss": row[2],
                        "Val Accuracy": row[3],
                    })
    return pd.DataFrame(data)


def plot_results(df):
    grouped = df.groupby("Config ID")
    
    for config_id, group in grouped:
        batch_size = group["Batch Size"].iloc[0]
        learning_rate = group["Learning Rate"].iloc[0]
        dropout_rate = group["Dropout Rate"].iloc[0]
        
        # Gràfica de Train Loss i Validation Loss
        plt.figure(figsize=(10, 6))
        plt.plot(group["Epochs"], group["Train Loss"], label="Train Loss", color="blue")
        plt.plot(group["Epochs"], group["Val Loss"], label="Validation Loss", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Train vs Validation Loss\nConfig ID: {config_id}, Batch Size: {batch_size}, LR: {learning_rate}, Dropout: {dropout_rate}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Gràfica de Validation Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(group["Epochs"], group["Val Accuracy"], label="Validation Accuracy", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy (%)")
        plt.title(f"Validation Accuracy\nConfig ID: {config_id}, Batch Size: {batch_size}, LR: {learning_rate}, Dropout: {dropout_rate}")
        plt.legend()
        plt.grid(True)
        plt.show()


# Exemple d'ús
file_path = "D:/Clase/UAB/TFG/results.csv"  # Ruta del fitxer
df = parse_results(file_path)

if not df.empty:
    plot_results(df)
else:
    print("No s'han trobat dades vàlides al fitxer.")
