import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset desde un archivo Excel
file_path = '/content/drive/MyDrive/Proyecto de Grado/DataSedPCA_FINAL3.xlsx'
df = pd.read_excel(file_path)

# Suponiendo que la última columna es la etiqueta (target)
X = df.iloc[:, :-1]  # Características (features)
y = df.iloc[:, -1]   # Etiquetas (labels)

# Usar train_test_split para muestreo estratificado
X_reduced, _, y_reduced, _ = train_test_split(X, y, stratify=y, test_size=0.7)

# Combinar de nuevo las características y las etiquetas en un DataFrame
df_reduced = pd.concat([X_reduced, y_reduced], axis=1)

# Guardar el dataset reducido a un nuevo archivo Excel
reduced_file_path = '/content/drive/MyDrive/Proyecto de Grado/DataSedPCA_FINAL3_Reducido.xlsx'
df_reduced.to_excel(reduced_file_path, index=False)

print("Dataset reducido guardado en:", reduced_file_path)