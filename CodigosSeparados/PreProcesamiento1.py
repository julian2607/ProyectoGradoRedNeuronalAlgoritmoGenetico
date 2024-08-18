import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# EN ESTE CODIGO SE ESTUDIA CUALES CON LAS COLUMANS CON MAYOR INFORMACION
# Y HACER LA REDUCCION DE DIMENSIONALIDAD EN BASE AL DATASET CON LA MINIMA
# CANTIDAD DE COLUMANS Y CON UN 95% DE INFORMACION


# PREPROSESAMIENTO DE LOs DATOS
# Cargar DataSet Y Eliminar duplciados
data= pd.read_excel('Proyecto de Grado\DataSetPruebas.xlsx')
data = data.drop_duplicates()

# Codificar variables categóricas Solo para el entrenamiento
# data = pd.get_dummies(data,columns=['label'])

# Columnas de entradas y salida
X = data.iloc[:, :-1]  # Todas las columnas excepto la última
y = data.iloc[:, -1]   # Última columna (variable objetivo)

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de Random Forest para evaluar la importancia de las características
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Obtener la importancia de las características
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Mostrar características más importantes
print("Características más importantes:")
for i in range(len(importances)):
    print(f"{data.columns[indices[i]]}: {importances[indices[i]]}")

# Graficar la importancia de las características
plt.figure(figsize=(15, 8))
plt.title("Importancia de las características Del dataset")
plt.xlabel("Columnas")
plt.ylabel("Normalización Importancia Datos")
plt.bar(range(len(importances)), importances[indices], align='center',color='green')
plt.xticks(range(len(importances)), data.columns[indices], rotation=90)
plt.savefig('Proyecto de Grado\static\Imagenes\ImagenesProcesadas\CaracteristicasImport.png', bbox_inches='tight')

# -------------------------------------------------------------------------------------------------
# Aplicar tecnica de reduccion de dimensionalidad PCA

# data= pd.read_excel('/content/drive/MyDrive/Proyecto de Grado/DataSet Intrusion IOT.xlsx')
data= pd.read_excel('Proyecto de Grado/DataSet Intrusion IOT.xlsx')
data = data.drop_duplicates()

# Eliminar Columnas no necesarias
columnas_a_eliminar = ['Telnet','IRC', 'SSH', 'DNS','DHCP','HTTP','Drate','cwr_flag_number','ece_flag_number','SMTP','DHCP','Drate','ARP','IPv','LLC','HTTPS']
data = data.drop(columns=columnas_a_eliminar)

# Columnas de entradas y salida
X1 = data.iloc[:, :-1]  # Todas las columnas excepto la última
y1 = data.iloc[:, -1]   # Última columna (variable objetivo)

scaler = StandardScaler()  # O MinMaxScaler() si prefieres normalización
X_scaled = scaler.fit_transform(X1)

# Verificar si X es un array de numpy o un DataFrame de pandas válido y no está vacío
if isinstance(X1, (np.ndarray, pd.DataFrame)) and X1.size > 0:        
    # Saber a que cantidad de dimensiones es correcto hacer
    pca1 = PCA()
    pca1.fit(X1)
    cumsum = np.cumsum(pca1.explained_variance_ratio_)
    NumComp = np.argmax(cumsum >= 0.95) + 1
    print("Cantidad de dimensiones al 95%: "+ str(NumComp))
    
    # Crear columnas para el nuevo DATASET
    Columnas=[]
    for D  in range(0,NumComp):
        Columnas.append('Columna' + str(D+1))

    # Ejecutar Reduccion de dimensionalidad
    print("Ejecutando PCA con " + str(NumComp)+ " Componentes")
    pca = PCA(n_components= NumComp)        
    X_reduced = pca.fit_transform(X1)   #explicar todo esto y documentar 
    pca.components_    
    pca.explained_variance_ratio_

    #Devolver a los datos originales
    X_inv = pca.inverse_transform(X_reduced)
    Diferecnia = np.mean(np.sum(np.square(X_inv - X1), axis=1))

    # Calcular la informacion perdida en la Reduccion de D    
    Diferecnia = 1 - pca.explained_variance_ratio_.sum()
    print("Perdidas de informacion: "+ str(Diferecnia))
    # Crear un nuevo DataFrame con las características reducidas y la variable objetivo
    print("Creando Documento")
    reduced_data = pd.DataFrame(X_reduced, columns=Columnas)
    reduced_data['Tipo de Ataque'] = y1.values
    reduced_data.to_excel('Proyecto de Grado/DatosProcesados/DataSedPCA_Procesado.xlsx', index=False)
    print("Documento Creado")
    
    # División del dataset con las características reducidas
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_reduced, y1, test_size=0.2, random_state=42)
    # Guardado del dataset procesado 
    np.save('Proyecto de Grado/DatosProcesados/X_train_pca.npy', X_train_pca)
    np.save('Proyecto de Grado/DatosProcesados/X_test_pca.npy', X_test_pca)
    np.save('Proyecto de Grado/DatosProcesados/y_train.npy', y_train)
    np.save('Proyecto de Grado/DatosProcesados/y_test.npy', y_test)

else:
    print("X no es un array de numpy o un DataFrame de pandas válido o está vacío.")

