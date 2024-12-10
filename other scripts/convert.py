import os
import time
import numpy as np
from tensorflow.keras.models import load_model

# Caminho dos dados e modelos
data_path = "./data"
models = {
    "distillation_model.h5": "distillation_model_results.txt",
    "distillation_model.keras": "distillation_model_keras_results.txt",
    "full_cnn_model.h5": "full_cnn_model_results.txt",
    "ultra_light_distillation_model.h5": "ultra_light_distillation_model_results.txt",
    "ultra_light_distillation_model.keras": "ultra_light_distillation_model_keras_results.txt"
}

# Diretório de saída para logs
save_path = "./inference_results/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Carregar dados

X_test = np.load(os.path.join(data_path, "X_test_Driving_NewApproach_Injected_v2.npz"))['arr_0']

Y_test = np.load(os.path.join(data_path, "Y_test_Driving_NewApproach_Injected_v2.npz"))['arr_0']

# Número de amostras aleatórias
#num_samples = 100000
#random_indices = np.random.choice(X_test.shape[0], size=num_samples, replace=False)
X_sample = X_test

# Iterar sobre os modelos e calcular o tempo de inferência
for model_file, result_file in models.items():
    if model_file.endswith(".keras"):
        pass
    else:
        try:
            # Carregar o modelo
            model = load_model(model_file)

            # Abrir arquivo para salvar os resultados
            with open(os.path.join(save_path, result_file), "a") as file:
                # Medir o tempo de inferência
                start = time.time()
                y_pred_scores = model.predict(X_sample, batch_size=1)
                end = time.time()

                # Calcular métricas de tempo
                total_time = end - start
                time_per_sample = total_time / y_pred_scores.shape[0]

                # Salvar os resultados no arquivo
                file.write(f"Modelo: {model_file}\n")
                file.write(f"Tempo total de inferência: {total_time:.6f} segundos\n")
                file.write(f"Tempo por amostra: {time_per_sample * 1e6:.2f} µs/amostra\n")
                file.write("=" * 40 + "\n")

            print(f"Resultados salvos para {model_file}.")

        except Exception as e:
            print(f"Erro ao processar o modelo {model_file}: {e}")
