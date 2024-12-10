import numpy as np

# Carregar os dados originais
x_val = np.load("processado/X_test_Driving_NewApproach_Injected_v2.npz")
x_val = x_val.f.arr_0
y_val = np.load("processado/Y_test_Driving_NewApproach_Injected_v2.npz")
y_val = y_val.f.arr_0

# Determinar o tamanho do novo conjunto (1% dos dados originais)
subset_size = max(1000, int(len(x_val) * 0.005))  # Pelo menos 1 amostra

# Selecionar uma amostra aleatória
indices = np.random.choice(len(x_val), subset_size, replace=False)

# Criar os novos datasets reduzidos
x_val_small = x_val[indices]
y_val_small = y_val[indices]

# Salvar os novos datasets
np.savez("X_test_small.npz", x_val_small)
np.savez("Y_test_small.npz", y_val_small)

print(f"Conjuntos reduzidos salvos com {subset_size} amostras:")
print(f"X_test_small: {x_val_small.shape}")
print(f"Y_test_small: {y_val_small.shape}")
