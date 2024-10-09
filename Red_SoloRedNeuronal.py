# -- coding: utf-8 --
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import joblib
import warnings
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam,Nadam,RMSprop
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Ignorar las advertencias de tensorflow
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Función para cargar tu dataset y dividirlo en características (X) y etiquetas (y)
def load_dataset():    
    # data = pd.read_excel('Proyecto de Grado\DataSedPCA.xlsx', skiprows=1)
    # leer dataset
    # data = pd.read_excel('Proyecto de Grado\DataSedPCA_FINAL1.xlsx')

    # # Valores a eliminar
    # valores_a_eliminar = [
    #     'Uploading_attack','Recon-PingSweep','XSS',
    #     'Backdoor_Malware','CommandInjection','SqlInjection',
    #     'BrowserHijacking','DictionaryBruteForce',
    #     'DDoS-SlowLoris','DDoS-HTTP_Flood',
    #     'VulnerabiliFyScan','DoS-HTTP_Flood',
    #     'Recon-OSScan','Recon-HostDiscovery',
    #     'DNS_Spoofing','DDoS-UDP_Fragmentation',
    #     'DDOS_ASK_Frafmetation','MITM-ArpSpoofing','DDoS-ICMP_Fragmentation',
    #     'Mirai-greip_flood','DoS-HTTP_Flood','DDoS-ICMP_Fragmentation'
    #     # 'BenignTraffic','MITM-ArpSpoofing','Mirai-udpplain',
    #     # 'Mirai-greeth_flood','DoS-SYN_Flood','DoS-TCP_Flood',
    #     # 'VulnerabilityScan','BenignTraffic','Uploading_Attack',
    #     # 'Recon-PortScan','Mirai-udpplain','Mirai-greip_flood',
    #     # 'Mirai-greeth_flood','DoS-UDP_Flood','DoS-TCP_Flood',
    #     #  'DoS-SYN_Flood','DDoS-ACK_Fragmentation'
    # ]

    # # Eliminar filas donde la columna tenga cualquiera de los valores especificados
    # data = data[~data['Tipo de Ataque'].isin(valores_a_eliminar)]
    # # Eliminar la primera fila
    # data = data.drop(index=1)
    # # Guardar Nuevo dataset
    # data.to_excel('Proyecto de Grado/DataSetSinValores.xlsx', index=False)
    # print("Documento Creado")

# ----------------------EJECUTAR RED NEURONAL CON DATOS PROCESADOS -----------------------------#
    data = pd.read_excel('Dataset\DataSedPCA_FINAL2 _PRUEBA.xlsx')
    inputs = data.iloc[:, :-1]  # Todas las columnas excepto la última
    Outputs = data.iloc[:, -1]   # Última columna (variable objetivo)

    # Codificar las etiquetas de salida usando One-Hot Encoding
    encoder = OneHotEncoder(sparse=False)
    Outputs = encoder.fit_transform(Outputs.values.reshape(-1, 1))

    # Normalizar las columnas de entrada
    scaler = MinMaxScaler()
    inputs= scaler.fit_transform(inputs)
    x_train, x_test, y_train, y_test = train_test_split(inputs, Outputs, test_size=0.3, random_state=42)
    
    # One-hot encoding para las etiquetas de salida con handle_unknown='ignore'
    # encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) 

    # Guardar los datos en archivos .npy
    np.save('CodigoGit/DatosProcesados/x_train.npy', x_train)
    np.save('CodigoGit/DatosProcesados/x_test.npy', x_test)
    np.save('CodigoGit/DatosProcesados/y_train.npy', y_train)
    np.save('CodigoGit/DatosProcesados/y_test.npy', y_test)

    # Guardar el OneHotEncoder ajustado
    joblib.dump(encoder, 'CodigoGit/DatosProcesados/encoder.joblib')
    joblib.dump(scaler, 'CodigoGit/DatosProcesados/scaler.joblib')
    return x_train, x_test, y_train, y_test

# ------------------------------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
if __name__ == "__main__":   

    # Cargar los datos de entrenamiento
    x_train, x_test, y_train, y_test = load_dataset()

    # Crear la red neuronal
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=x_train.shape[1], kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))  # Usa softmax para clasificación multiclase
    # Compilar el modelo
    optimizer = Adam(learning_rate=0.001)  # Ajustar la tasa de aprendizaje
    # ['accuracy','Precision', 'Recall', 'AUC','f1_score']
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy','Precision'])

    # Definir el early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

    # Entrenar la red neuronal
    History = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2,callbacks=[early_stopping])

    
    # Otras métricas (si están disponibles):
    accuracy = History.history['accuracy']
    val_accuracy = History.history['val_accuracy']
    precision = History.history['Precision']
    # recall = History.history['Recall']
    # f1_score = History.history['f1_score']

    # Crear un DataFrame con las métricas
    metrics_df = pd.DataFrame({
        'Epoch': range(1, len(accuracy) + 1),
        'Accuracy': accuracy,
        'Validation Accuracy': val_accuracy,
        'Loss': History.history['loss'],
        'Validation Loss': History.history['val_loss'],
        'precision': precision
    })

    #Guardar informacion del entrenamiento en 
    print("Guardando Historial Entrenamiento")
    metrics_df.to_csv('CodigoGit/DatosProcesados/historial_entrenamiento.csv', index=False)

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy, loss2 = model.evaluate(x_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Evaluar el modelo en el conjunto de prueba
    y_pred = model.predict(x_test)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
    print(f'Exactitud: {accuracy}')
    print(f'Precisión: {precision}')
    print(f'Sensibilidad: {recall}')


    # Evaluar en el conjunto de entrenamiento
    train_loss, train_accuracy,train_loss2 = model.evaluate(x_train, y_train)
    print(f'Training Loss: {train_loss}, Training Accuracy: {train_accuracy}')

    # Guardar el modelo y el escalador
    print("Creando documento Modelo")
    model.save('CodigoGit/DatosProcesados/modelo.h5')
    print("Documento Modelo Red Creado")

    #------------------------ Mostrar Resulatdios y graficas del entrennamiento.-----------------------#
    # Cargar el archivo CSV
    df = pd.read_csv("CodigoGit/DatosProcesados/historial_entrenamiento.csv")
    # Gráfico de precisión
    plt.subplot(2, 1, 1)
    plt.plot(df['Epoch'], df['Accuracy'], label='Precisión (Training)')
    plt.plot(df['Epoch'], df['Validation Accuracy'], label='Precisión (Validación)')
    plt.xlabel('Epochs')
    plt.ylabel('Precisión')
    plt.legend()

    # Gráfico de pérdida
    plt.subplot(2, 1, 2)
    plt.plot(df['Epoch'], df['Loss'], label='Pérdida (Training)')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Pérdida (Validación)')
    plt.xlabel('Epochs')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.tight_layout()    
    plt.savefig('CodigoGit/static/ImagenesProcesadas/EstadisticasEntrenamiento.png', bbox_inches='tight')
    

