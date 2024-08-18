import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Cargar el modelo desde el archivo .h5
model = load_model('Proyecto de Grado\/modelo.h5')

# Cargar los datos desde los archivos .npy
x_train = np.load('Proyecto de Grado/DatosProcesados/x_train.npy')
x_test = np.load('Proyecto de Grado/DatosProcesados/x_test.npy')
y_train = np.load('Proyecto de Grado/DatosProcesados/y_train.npy')
y_test = np.load('Proyecto de Grado/DatosProcesados/y_test.npy')

# cargar encoder 
encoder = joblib.load('Proyecto de Grado/DatosProcesados/encoder.joblib')
scaler = joblib.load('Proyecto de Grado/DatosProcesados/scaler.joblib')

# Asumiendo que tienes nuevas entradas y etiquetas
data = pd.read_excel('Proyecto de Grado\DataSedPCA_FINAL2 _PRUEBA.xlsx')
new_inputs = data.iloc[:, :-1]  # Todas las columnas excepto la última
new_outputs = data.iloc[:, -1]  # Última columna (variable objetivo)
# Normaliza las nuevas entradas usando el scaler cargado
new_inputs = scaler.transform(new_inputs)
# Codifica las etiquetas de salida usando el encoder cargado
new_outputs = encoder.transform(new_outputs.values.reshape(-1, 1))



# Evaluar el modelo en los datos de prueba procesados adecuadamente
# evaluation = model.evaluate(x_test, y_test)
evaluation = model.evaluate(new_inputs, new_outputs)
loss = evaluation[0]
accuracy = evaluation[1]
print(f'Pérdida: {loss}, Exactitud: {accuracy}')

# Obtener predicciones y convertir a formato binario si es necesario
y_pred_prob = model.predict(x_test)
y_test_binary = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
y_pred_binary = np.argmax(y_pred_prob, axis=1)

# Verificar los tamaños de y_test_binary y y_pred_binary
print(f'Tamaño de y_test_binary: {y_test_binary.shape}')
print(f'Tamaño de y_pred_binary: {y_pred_binary.shape}')

# Asegurarse de que los tamaños coincidan
if y_test_binary.shape != y_pred_binary.shape:
    raise ValueError("El tamaño de y_test_binary y y_pred_binary no coincide.")

# Calcular métricas adicionales
precision = precision_score(y_test_binary, y_pred_binary, average='macro')
recall = recall_score(y_test_binary, y_pred_binary, average='macro')
f1 = f1_score(y_test_binary, y_pred_binary, average='macro')
auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr') if y_test.ndim > 1 else roc_auc_score(y_test, y_pred_prob)

print(f'Precision: {precision}, Sencibilidad: {recall}, F1-Score: {f1}, AUC: {auc}')

# Graficar la matriz de confusión
cm = confusion_matrix(y_test_binary, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Matriz de Confusión')
plt.show()

# # Evaluar el modelo en los datos de prueba
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f'Pérdida: {loss}, Precisión: {accuracy}')

# # Obtener las predicciones del modelo
# y_pred_prob = model.predict(x_test)
# y_pred = np.argmax(y_pred_prob, axis=1)  # Convertir probabilidades a clases

# # Calcular métricas adicionales
# average_type = 'macro'  # Puede ser 'micro', 'macro', 'weighted', 'samples'
# precision = precision_score(y_test, y_pred, average=average_type)
# recall = recall_score(y_test, y_pred, average=average_type)
# f1 = f1_score(y_test, y_pred, average=average_type)
# auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

# print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}, AUC: {auc}')

# # Graficar la matriz de confusión
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.title('Matriz de Confusión')
# plt.show()

# # Generar los gráficos de métricas
# metrics = {
#     'Pérdida': loss,
#     'Precisión': accuracy,
#     'Precision': precision,
#     'Recall': recall,
#     'F1-Score': f1,
#     'AUC': auc
# }

# # Crear un gráfico de barras para las métricas
# plt.figure(figsize=(10, 5))
# plt.bar(metrics.keys(), metrics.values())
# plt.title('Métricas del Modelo')
# plt.xlabel('Métrica')
# plt.ylabel('Valor')
# plt.show()

# # Graficar la matriz de confusión
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.title('Matriz de Confusión')
# plt.show()