#Run in Version "out"
import tensorflow as tf
import json
import numpy as np

path_rec = "/home/pedro/projetoDL/saved models/h5/to_convert/ultra_distilattion/"
weights_path = path_rec + "weights.npz"
architecture_path = path_rec + "architecture.json"

# Carregar a arquitetura
with open(architecture_path, "r") as arch_file:
    architecture = json.load(arch_file)

# Criar o modelo sequencial do zero
model = tf.keras.Sequential()

for layer_config in architecture:
    class_name = layer_config["class_name"]
    config = layer_config["config"]

    # Ajustar dtype para float32
    if "dtype" in config and isinstance(config["dtype"], dict):
        dtype_config = config["dtype"]
        if dtype_config.get("class_name") == "DTypePolicy":
            print(f"Convertendo DTypePolicy para float32 na camada {config['name']}.")
            config["dtype"] = "float32"

    # Remover atributos não suportados
    config.pop("dtype_policy", None)
    config.pop("synchronized", None)

    try:
        # Recriar a camada com configurações ajustadas
        layer = getattr(tf.keras.layers, class_name).from_config(config)
        model.add(layer)
    except Exception as e:
        print(f"Erro ao recriar camada {class_name}: {e}")
        raise e

# Inicializar os pesos chamando build() com a entrada correta
model.build(input_shape=(None, 44, 116, 1))  # Batch size flexível e entrada 44x116x1

# Carregar os pesos
weights_data = np.load(weights_path, allow_pickle=True)
weights = [weights_data[key] for key in weights_data.files]

try:
    model.set_weights(weights)
    print("Pesos carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar pesos: {e}")
    raise e

# Salvar o modelo reconstruído
model.save(path_rec + "full_cnn_tf241.h5")
print("Modelo reconstruído e salvo com sucesso!")
