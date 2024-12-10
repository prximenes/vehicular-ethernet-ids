import os
import tensorflow as tf

# Caminho dos modelos e pasta de saída
model_dir = "."
output_dir = "models_lite"

# Criar pasta de saída, se não existir
os.makedirs(output_dir, exist_ok=True)

# Listar arquivos de modelos no diretório atual
model_files = [f for f in os.listdir(model_dir) if f.endswith(('.h5', '.keras'))]

# Função para converter e salvar modelos em TFLite
def convert_to_tflite(model_path, output_path):
    try:
        # Carregar o modelo
        model = tf.keras.models.load_model(model_path)
        # Converter para TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        # Salvar modelo TFLite
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"Modelo convertido com sucesso: {output_path}")
    except Exception as e:
        print(f"Erro ao converter o modelo {model_path}: {e}")

# Converter cada modelo encontrado
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    output_path = os.path.join(output_dir, model_file.rsplit('.', 1)[0] + ".tflite")
    convert_to_tflite(model_path, output_path)

print(f"Conversão concluída! Modelos TFLite salvos em: {output_dir}")
