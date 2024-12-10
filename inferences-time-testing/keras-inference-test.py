import numpy as np
import tensorflow as tf
import time
from distillation_model import DistillationModel  # Importar a classe personalizada

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
    print("1 - distillation_model.keras")
    print("2 - ultra_light_distillation_model.keras")

    escolha = input("Digite o número do modelo desejado: ")
    if escolha == "1":
        return "../saved models/keras/distillation_model.keras"
    elif escolha == "2":
        return "../saved models/keras/ultra_light_distillation_model.keras"
    elif escolha == "0":
        return ""
    else:
        print("Escolha inválida. Tente novamente.")
        return escolher_modelo()

# Escolher o modelo
modelo_escolhido = escolher_modelo()

# Carregar os dados
x_val = np.load("../dataset/reduced-to-rasp/small_x_val.npy")
y_val = np.load("../dataset/reduced-to-rasp/small_y_val.npy")

# Garantir que os dados estejam no formato correto
x_val = np.expand_dims(x_val, axis=-1).astype(np.float32)  # Adicionar a dimensão do canal se necessário

# Caso o modelo seja Keras
print(f"Usando modelo Keras: {modelo_escolhido}")
# Carregar o modelo .keras com suporte à classe personalizada
model = tf.keras.models.load_model(
    modelo_escolhido,
    custom_objects={"DistillationModel": DistillationModel}
)

# Medir o tempo de inferência
start_time = time.time()

# Realizar a inferência diretamente
y_pred_scores = model.predict(x_val, batch_size=1)

end_time = time.time()

# Tempo total e por amostra
total_time = end_time - start_time
time_per_sample = total_time / len(x_val)

print(f"Tempo total de inferência: {total_time:.4f} segundos")
print(f"Tempo por amostra: {time_per_sample * 1e6:.2f} µs")
