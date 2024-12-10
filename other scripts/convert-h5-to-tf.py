import numpy as np
import tensorflow as tf
import time, os

# Verificar se há GPUs disponíveis
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU disponível. O código será executado na GPU.")
    for gpu in gpus:
        print(f"Dispositivo GPU encontrado: {gpu}")
else:
    print("GPU não disponível. O código será executado na CPU.")

# Função para escolher o modelo
def escolher_modelo():
    print("Escolha um modelo:")
    print("1 - full_cnn_model.h5")
    print("2 - distillation_model.h5")
    print("3 - ultra_light_distillation_model.h5")

    escolha = input("Digite o número do modelo desejado: ")
    if escolha == "1":
        return "../saved models/h5/full_cnn_model.h5"
    elif escolha == "2":
        return "../saved models/h5/distillation_model.h5"
    elif escolha == "3":
        return "../saved models/h5/ultra_light_distillation_model.h5"
    else:
        print("Escolha inválida. Tente novamente.")
        return escolher_modelo()

# Escolher o modelo
modelo_escolhido = escolher_modelo()

output_model_path = os.path.splitext(modelo_escolhido)[0] + '_adjusted_model.h5'

# Caso o modelo seja H5
print(f"Usando modelo H5: {modelo_escolhido}")
# Carregar o modelo H5
model = tf.keras.models.load_model(modelo_escolhido)

# Obter a configuração do modelo
model_config = model.get_config()

# Ajustar as camadas para remover o 'batch_shape' e reintroduzir 'shape' na InputLayer
adjusted_layers = []
for layer_config in model_config['layers']:
    if layer_config['class_name'] == 'InputLayer':
        if 'batch_shape' in layer_config['config']:
            print(f"Removendo 'batch_shape' da camada {layer_config['class_name']}")
            batch_shape = layer_config['config'].pop('batch_shape')
            # Adicionar 'shape' com base no 'batch_shape'
            layer_config['config']['shape'] = batch_shape[1:]  # Remove o batch size
    adjusted_layers.append(tf.keras.layers.deserialize(layer_config))

# Criar um novo modelo sequencial (ou funcional) a partir das camadas ajustadas
adjusted_model = tf.keras.Sequential(layers=adjusted_layers)

# Copiar os pesos do modelo original para o ajustado
adjusted_model.set_weights(model.get_weights())

# Salvar o modelo ajustado no formato HDF5
adjusted_model.save("/home/pedro/projetoDL/other scripts/adjusted_model.h5")
