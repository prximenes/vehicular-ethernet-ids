import numpy as np
import tensorflow as tf
import time

# Função para escolher o modelo
def escolher_modelo():
    print("Escolha um modelo:")
    print("1 - 10xPrunned_TFLite.tflite")
    print("2 - TFLite_3xPrunned.tflite")
    print("3 - model_pruned_gpu.h5")
    escolha = input("Digite o número do modelo desejado: ")
    if escolha == "1":
        return "10xPrunned_TFLite.tflite"
    elif escolha == "2":
        return "TFLite_3xPrunned.tflite"
    elif escolha == "3":
        return "model_pruned_gpu.h5"
    else:
        print("Escolha inválida. Tente novamente.")
        return escolher_modelo()

# Escolher o modelo
modelo_escolhido = escolher_modelo()

# Carregar os dados
x_val = np.load("small_x_val.npy")
y_val = np.load("small_y_val.npy")

# Garantir que os dados estejam no formato correto
x_val = np.expand_dims(x_val, axis=-1).astype(np.float32)  # Adicionar a dimensão do canal se necessário

# Inferência usando TFLite ou H5
if modelo_escolhido.endswith(".tflite"):
    print(f"Usando modelo TFLite: {modelo_escolhido}")
    # Carregar o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=modelo_escolhido)
    interpreter.allocate_tensors()

    # Obter detalhes dos tensores de entrada e saída
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Medir o tempo de inferência
    start_time = time.time()

    y_pred_scores = []
    for sample in x_val:
        sample = np.expand_dims(sample, axis=0)  # Adicionar dimensão do batch
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        y_pred_scores.append(prediction)

    end_time = time.time()

else:  # Caso o modelo seja H5
    print(f"Usando modelo H5: {modelo_escolhido}")
    # Carregar o modelo H5
    model = tf.keras.models.load_model(modelo_escolhido)

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
