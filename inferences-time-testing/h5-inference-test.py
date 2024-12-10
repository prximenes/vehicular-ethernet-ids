import numpy as np
import tensorflow as tf
from keras.models import load_model
import time

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
    elif escolha == "0":
        return "../saved models/h5/ultra_light_distillation_adjusted_model.h5"
    else:
        print("Escolha inválida. Tente novamente.")
        return escolher_modelo()

# Escolher o modelo
modelo_escolhido = escolher_modelo()

# Carregar os dados
#x_val = np.load("../dataset/reduced-to-rasp/small_x_val.npy")
#y_val = np.load("../dataset/reduced-to-rasp/small_y_val.npy")

x_val = np.load("../dataset/processado/X_test_small.npz")
x_val = x_val.f.arr_0
y_val = np.load("../dataset/processado/Y_test_small.npz")
y_val = y_val.f.arr_0

# Garantir que os dados estejam no formato correto
#x_val = np.expand_dims(x_val, axis=-1).astype(np.float32)  # Adicionar a dimensão do canal se necessário

# Caso o modelo seja H5
print(f"Usando modelo H5: {modelo_escolhido}")
# Carregar o modelo H5
model = tf.keras.models.load_model(modelo_escolhido)

# Medir o tempo de inferência
start_time = time.time()

# Realizar a inferência diretamente
y_pred_scores = model.predict(x_val)

end_time = time.time()

# Tempo total e por amostra
total_time = end_time - start_time
time_per_sample = total_time / len(x_val)

print(f"Tempo total de inferência: {total_time:.4f} segundos")
print(f"Tempo por amostra: {time_per_sample * 1e6:.2f} µs")
