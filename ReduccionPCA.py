from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Cargar DataSet
data = pd.read_excel('Proyecto de Grado\DataSetPruebas.xlsx')
# Columnas de entradas y salida
X = data.iloc[:, :-1]  # Todas las columnas excepto la última
y = data.iloc[:, -1]   # Última columna (variable objetivo)

# Verificar si X es un array de numpy o un DataFrame de pandas válido y no está vacío
if isinstance(X, (np.ndarray, pd.DataFrame)) and X.size > 0:        
    # Saber a que cantidad de dimensiones es correcto hacer
    pca1 = PCA()
    pca1.fit(X)
    cumsum = np.cumsum(pca1.explained_variance_ratio_)
    NumComp = np.argmax(cumsum >= 0.95) + 1
    print("Cantidad de dimensiones al 95%: "+ str(NumComp))

    # Ejecutar Reduccion de dimensionalidad
    print("Ejecutando PCA con " + str(NumComp)+ " Componentes")
    pca = PCA(n_components= NumComp)        
    X_reduced = pca.fit_transform(X)    
    pca.components_    
    pca.explained_variance_ratio_

    #Devolver a los datos originales
    X_inv = pca.inverse_transform(X_reduced)
    Diferecnia = np.mean(np.sum(np.square(X_inv - X), axis=1))

    # Calcular la infromacion perdida en la Reduccion de D    
    Diferecnia = 1 - pca.explained_variance_ratio_.sum()
    print("Perdidas de informacion: "+ str(Diferecnia))
    # Crear un nuevo DataFrame con las características reducidas y la variable objetivo
    print("Creando Documento")
    reduced_data = pd.DataFrame(X_reduced, columns=['Componente 1', 'Componente 2','Componente 3','Componente 4','Componente 5','Componente 6'])
    reduced_data['Tipo de Ataque'] = y.values
    reduced_data.to_excel('Proyecto de Grado\DataSedPCA.xlsx', index=False)
    print("Documento Creado")
else:
    print("X no es un array de numpy o un DataFrame de pandas válido o está vacío.")
