import tensorflow as tf
import json
import numpy as np

path = "/home/pedro/projetoDL/saved models/h5"
# Carregar o modelo original
original_model = tf.keras.models.load_model(path + "/ultra_light_distillation_model.h5")

# Salvar a arquitetura do modelo
architecture = []
for layer in original_model.layers:
    layer_config = {
        "class_name": layer.__class__.__name__,
        "config": layer.get_config()
    }
    architecture.append(layer_config)

with open(path + "/to_convert/ultra_distilattion/architecture.json", "w") as arch_file:
    json.dump(architecture, arch_file, indent=4)

# Salvar os pesos do modelo
weights_path = path + "/to_convert/ultra_distilattion/weights.npz"
np.savez(weights_path, *original_model.get_weights())

print(f"Arquitetura salva em 'architecture.json' e pesos salvos em 'weights.npz'.")
