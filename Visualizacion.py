# -- coding: utf-8 --
# ------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import joblib
import io
import contextlib
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

# librerias red neuronal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam,Nadam,RMSprop
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# ------------------------------------------------------------
# Bibliotecas pagina web
import threading
import socket
import time
from tkinter import *
from tkinter import Tk
from tkinter import ttk
from flask import Flask, request, render_template,jsonify
# PDF
import pandas as pd
from fpdf import FPDF

# Importar librería del conector de mysql
import mysql.connector as mysql
# Importe la librería SMTP
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Iniciar Flask
Pagina = Flask(__name__)

# Variables Globales
global Dataset


# --------------------CARGAR DATOS PARA PRUEBAS ----------------------------------#
def Cargardatos():
    global Dataset, inputs, Outputs, x_train, x_test, y_train, y_test
    print("Cargando datos")
    data = Dataset
    inputs = data.iloc[:, :-1]  # Todas las columnas excepto la última
    Outputs = data.iloc[:, -1]   # Última columna (variable objetivo)
    x_train, x_test, y_train, y_test = train_test_split(inputs, Outputs, test_size=0.3, random_state=42)
    # Cargar el encoder y las clases
    all_classes = np.load('Proyecto de Grado\classes.npy', allow_pickle=True)
    encoder_categories = np.load('Proyecto de Grado\encoder_categories.npy', allow_pickle=True).tolist()
    encoder = OneHotEncoder(categories=encoder_categories, sparse=False)
    # Transformar las etiquetas de salida
    y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_test = encoder.transform(y_test.values.reshape(-1, 1))
    # Guardar los datos en archivos .npy
    np.save('Proyecto de Grado/DatosProcesados/x_train.npy', x_train)
    np.save('Proyecto de Grado/DatosProcesados/x_test.npy', x_test)
    np.save('Proyecto de Grado/DatosProcesados/y_train.npy', y_train)
    np.save('Proyecto de Grado/DatosProcesados/y_test.npy', y_test)
     
    # Dimenciones Datos pruebas
    print(x_test.shape)

#--------------------- DISTRIBUCION DE LOS DATOS ------------------------------------------#
def DistribucionDatos():
    global Dataset
    df1 = Dataset
    print("Distribucion de los datos")
    # Crear gráficos de barras para datos categóricos
    for column in df1.select_dtypes(include=['object']).columns:
        df1[column].value_counts().plot(kind='bar')
        plt.title(f'Distribución de los datos Columan: {column}')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
    plt.savefig('Proyecto de Grado\static\ImagenesProcesadas\CaracteristicasImport2.png')

    # Contar la cantidad de cada tipo de ataque
    ataques_counts = df1['label'].value_counts()
    # Calcular el porcentaje de cada tipo de ataque
    ataques_percentage = (ataques_counts / ataques_counts.sum()) * 100
    # Crear un DataFrame con el nombre, las cantidades y porcentajes
    result_df = pd.DataFrame({
        'Nombre': ataques_counts.index,
        'Cantidad': ataques_counts.values,
        'Porcentaje': ataques_percentage.values
    })
    # Guardar el DataFrame en un archivo Excel
    result_df.to_excel('Proyecto de Grado/DatosProcesados/Cantidad_y_Porcentaje_Ataques.xlsx', index=False)

    # Graficar un gráfico circular para el porcentaje de cada tipo de ataque
    plt.figure(figsize=(10, 8))
    plt.pie(ataques_percentage, labels=ataques_percentage.index, autopct='%1.1f%%', startangle=140)
    plt.title('Porcentaje de cada tipo de ataque')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('Proyecto de Grado\static\ImagenesProcesadas\Porcentaje_Ataques.png', bbox_inches='tight')

# -------------------------INFORMACION MUTUA ---------------------------------------------#
def InfromacionMutua():
    global Dataset, inputs, Outputs
    df=Dataset
    # Suponiendoque la última columna es la etiqueta y el resto son características
    X = inputs # Todas las columnas excepto la última
    y = Outputs # La última columna
    # Calcular la información mutua
    mi = mutual_info_classif(X, y, discrete_features='auto')
    # Crear un DataFrame con los resultados
    mi_df = pd.DataFrame(mi, index=X.columns, columns=['Mutual Information'])
    # Mostrar un gráfico de barras de la información mutua
    plt.figure(figsize=(10, 6))
    mi_df.sort_values(by='Mutual Information', ascending=False).plot(kind='bar', legend=None)
    plt.title('Información Mutua entre Características y Etiqueta')
    plt.xlabel('Características')
    plt.ylabel('Información Mutua')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()  # Ajustar el diseño para que no se corten las etiquetas
    plt.savefig('Proyecto de Grado\static\ImagenesProcesadas\ImportaciaCaractDatos.png', bbox_inches='tight')
    # plt.show()
    # Guardar los resultados en un archivo CSV
    # mi_df.to_csv('mutual_information_results.csv')
    print("Información mutua calculada y guardada en 'mutual_information_results.csv'.")

# -------------MATRIZ CORRELACION DATOS CON COLUMNA OBJETIVO-----------------------------#
def MatrizCorrelacion_1():
    print("Matriz de correlacion 1")
    global Dataset
    # Obtén el nombre de la última columna (columna categórica)
    categoria_col = Dataset.columns[-1]
    # Realiza la codificación one-hot de la columna categórica
    df_one_hot = pd.get_dummies(Dataset, columns=[categoria_col])
    # Calcular la matriz de correlación
    matriz_correlacion = df_one_hot.corr()
    # Filtrar la matriz de correlación para solo mostrar las correlaciones con las columnas categóricas codificadas
    categorical_columns = [col for col in df_one_hot.columns if col.startswith(categoria_col)]
    correlation_with_categorical = matriz_correlacion[categorical_columns].drop(index=categorical_columns)
    # Mostrar la matriz de correlación
    print(correlation_with_categorical)
    # Crear un mapa de calor para la matriz de correlación sin mostrar los valores
    plt.figure(figsize=(16,25))
    sns.heatmap(correlation_with_categorical, annot=False, cmap='coolwarm')
    plt.title('Matriz de Correlación')    
    plt.savefig('Proyecto de Grado\static\ImagenesProcesadas\MatrizCorrelacion.png', bbox_inches='tight')

#-------------------- MATRIZ DE CORRELACION 2--------------------------------------------#
def MatrizCorrelacion_2():
    global Dataset
    print("Matriz de correlacion 2")
    # Aplicar one-hot encoding a las columnas categóricas
    df_encoded = pd.get_dummies(Dataset, columns=['label'])
    # Calcular la matriz de correlación solo entre las características numéricas
    matriz_correlacion_numerica = df_encoded.corr()
    # Mostrar la matriz de correlación
    print(matriz_correlacion_numerica)
    # Crear un mapa de calor para la matriz de correlación sin mostrar los valores
    plt.figure(figsize=(16, 12))  # Ajusta el tamaño de la figura según sea necesario
    sns.heatmap(matriz_correlacion_numerica, annot=False, cmap='coolwarm', cbar=True)
    plt.title('Matriz de Correlación entre Características')
    # Rotar las etiquetas del eje x para que no se corten y queden verticales
    plt.xticks(rotation=90, ha='center')
    # Mostrar el gráfico
    plt.savefig('Proyecto de Grado\static\ImagenesProcesadas\MatrizCorrelacion2.png', bbox_inches='tight')

# -------------------------------PCA -------------------------------------------------------#
@Pagina.route('/ReduccionDimensionalidadPCA' , methods = ['GET', 'POST'])
def FuncionPCA():
    global Dataset,inputs,Outputs
    # DATASET VALIDAR SI ESTA CARGADO

    try:
        data = Dataset  
    except NameError:        
        # CARGAR INFORMACION DEL DATASET
        Url = "Proyecto de Grado\DataSetPruebas.xlsx"        
        print(f"Direccion: {Url}")
        Dataset = pd.read_excel(Url)

    data = Dataset      
    # OPTENER VALORES DEL FORMUALRIO    
    Valores_pagina = request.form 
    PORCENTAJE=Valores_pagina['PORCENTAJE_ELIMINAR']
    ColumnasELIMINAR=Valores_pagina['COLUMNAS_ELIMINAR']
    
    # VALIDAR ENTRADAS QUE NO SEAN VACIAS
    if ColumnasELIMINAR == "":
        columnas_a_eliminar = ['Telnet','IRC', 'SSH', 'DNS','DHCP','HTTP','Drate','cwr_flag_number','ece_flag_number','SMTP','DHCP','Drate','ARP','IPv','LLC','HTTPS']        
    else:
        columnas_a_eliminar = ColumnasELIMINAR
    # PORCENTAJE DESEADO
    if PORCENTAJE == "":
        PORCENTAJE=0.95
    else:
        PORCENTAJE = int(PORCENTAJE)/100

    print(f"Porcentaje: {PORCENTAJE}, Columnas a Elimnar: {columnas_a_eliminar}")
    
    try:        
        # ELIMNAR COLUMNAS NO DESEADAS        
        data = data.drop(columns=columnas_a_eliminar)
    except ZeroDivisionError as e:
        # Código para manejar la excepción
        print(f"ERROR ELIMINANDO COLUMNAS DEL DATASET ERROR: {e}")
    finally:
        print("Proceso de eliminar columans finalizados")

    # REALIZAR PCA
    X = data.iloc[:, :-1]  # Todas las columnas excepto la última
    y = data.iloc[:, -1]   # Última columna (variable objetivo)
    
    # Verificar si X es un array de numpy o un DataFrame de pandas válido y no está vacío
    if isinstance(X, (np.ndarray, pd.DataFrame)) and X.size > 0:
        # Saber a que cantidad de dimensiones es correcto hacer
        pca1 = PCA()
        pca1.fit(X)
        cumsum = np.cumsum(pca1.explained_variance_ratio_)
        NumComp = np.argmax(cumsum >= PORCENTAJE) + 1
        print("Cantidad de dimensiones al 95%: "+ str(NumComp))

        # Crear columnas para el nuevo DATASET
        Columnas=[]
        for D  in range(0,NumComp):
            Columnas.append('Columna' + str(D+1))

        # Ejecutar Reduccion de dimensionalidad
        print("Ejecutando PCA con " + str(NumComp)+ " Componentes")
        pca = PCA(n_components= NumComp)
        X_reduced = pca.fit_transform(X)
        pca.components_
        pca.explained_variance_ratio_

        #Devolver a los datos originales
        X_inv = pca.inverse_transform(X_reduced)
        Diferecnia = np.mean(np.sum(np.square(X_inv - X), axis=1))

        # Calcular la informacion perdida en la Reduccion de D
        Diferecnia = 1 - pca.explained_variance_ratio_.sum()
        print("Perdidas de informacion: "+ str(Diferecnia))
        # Crear un nuevo DataFrame con las características reducidas y la variable objetivo
        print("Creando Documento")
        reduced_data = pd.DataFrame(X_reduced, columns=Columnas)
        reduced_data['Tipo de Ataque'] = y.values
        reduced_data.to_excel('Proyecto de Grado/DatosProcesados/DataSedPCA_PROCESADO.xlsx', index=False)
        print("Documento Creado")

        # División del dataset con las características reducidas
        X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
        # Guardado del dataset procesado
        np.save('Proyecto de Grado/DatosProcesados/X_train_pca.npy', X_train_pca)
        np.save('Proyecto de Grado/DatosProcesados/X_test_pca.npy', X_test_pca)
        np.save('Proyecto de Grado/DatosProcesados/y_train.npy', y_train)
        np.save('Proyecto de Grado/DatosProcesados/y_test.npy', y_test)

         # Comando guaradr ganador tabla
        BD = mysql.connect(host=ORIGEN, user=USUARIO, passwd=CONTRASENA, db=BASEDATOS)
        Cursor = BD.cursor()
        Comando="insert into ReduccionPCA(Porcentaje,Numero_dimenciones,Perdida_Info) values(%s, %s,%s);"        
        Valores=(str(PORCENTAJE*100),str(NumComp),str(Diferecnia))
        Cursor.execute(Comando,Valores)  
        BD.commit()          

        # CONSULTAR INFROMACION DE LOS PCA        
        Cursor.execute("SELECT * FROM ReduccionPCA")
        MSG = MIMEMultipart()
        html_table=""

        for row in Cursor:
            html_table += "<tr>"
            for value in row:
                html_table += "<td>{}</td>".format(value)
            html_table += "</tr>"

        #CERRAR CONEXION 
        BD.commit()               
        BD.close()
        return jsonify({'TablaPCA': html_table})
    
    else:
        print("X no es un array de numpy o un DataFrame de pandas válido o está vacío.")

# ---------------------- ENTRENAR SOLOMODELO-------------------------------------------------#
def modeloRed(Epocas,bach,Earling,validation):
    global Dataset, inputs, Outputs, x_train, x_test, y_train, y_test
    
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
    History = model.fit(x_train, y_train, epochs=Epocas, batch_size=bach, validation_split=0.2,callbacks=[early_stopping])

    
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
    metrics_df.to_csv('Proyecto de Grado/historial_entrenamiento.csv', index=False)

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
    model.save('Proyecto de Grado/modelo.h5')
    print("Documento Modelo Red Creado")

    #------------------------ Mostrar Resulatdios y graficas del entrennamiento.-----------------------#
    # Cargar el archivo CSV
    df = pd.read_csv("Proyecto de Grado/historial_entrenamiento.csv")
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
    plt.savefig('Proyecto de Grado/static/EstadisticasEntrenamiento.png', bbox_inches='tight')
    
    # GUARDAR INFORMACION EN BASE DE DATOS
    BD = mysql.connect(host=ORIGEN, user=USUARIO, passwd=CONTRASENA, db=BASEDATOS)
    Cursor = BD.cursor()
    Comando="insert into RedNeuronal(Epocas,Batch,Validacion,Training_loss,loss,Training_Acurrancy,Exactitud,Presicion,Sensibilidad) values(%s, %s,%s,%s, %s,%s,%s,%s,%s);"        
    Valores=(str(Epocas),str(bach),str(validation),str(train_loss),str(loss),str(train_accuracy),str(accuracy),str(precision),str(recall))
    Cursor.execute(Comando,Valores)  
    BD.commit()  
  

# ------------------- ENTRENAR MODELO CON ALGORITMO-----------------------------------------#
def modeloAlgoritmo():
    global Dataset, inputs, Outputs, x_train, x_test, y_train, y_test


# ------------------------CARGAR INFORMACION DEL MODELO------------------------------------#
#-------------- Este metodo llama las fuciones del preprocesamiento------------------------#
@Pagina.route('/PreProcesamiento' , methods = ['GET', 'POST'])
def cargarinfo():
    global Dataset,EpocasRed,batch_size,validation_split,early_stopping
    print("Realizando Preprocesamiento")
    # CARGAR VALORES FORMULARIO
    Valores_pagina = request.form 
    Url=Valores_pagina['UrlDatasetPre']
    
    #RUTA DATOSPREPROCESADPS
    # "Proyecto de Grado\DataSedPCA _PUEBAS.xlsx"   

    # CARGAR INFORMACION DEL DATASET
    if Url == "": 
        Url = "Proyecto de Grado\DataSetPruebas.xlsx"        
    print(f"Direccion: {Url}")
    
    try:
        df = pd.read_excel(Url)
        Dataset = df
        print("DatasSet Cargado")

        # CARGAR FUNCIONES
        DistribucionDatos()
        Cargardatos()
        MatrizCorrelacion_1()
        MatrizCorrelacion_2()
        InfromacionMutua()
        
    except ZeroDivisionError as e:
        # Código para manejar la excepción
        print(f"Ruta No valida: {e}")
    finally:        
        print("Fin Cargue de datos.")
     
    # Segunda Correlacion
    return render_template('ReduccionDimensionalidad.html')

# ------------------------ENTRENAR MODELO RED CON O SIN ALGORTIMO------------------------------------#
# ----------- Este metodo llama las fuciones PARA ENTRENAR EL MODELO CON EL ALGORTIMO----------#
@Pagina.route('/EntrenarModelo' , methods = ['GET', 'POST'])
def EntrenarModelo_Resultado():

    print("Entrenando modelo")
    Valores_pagina = request.form 

    # indicador
    Indicador = Valores_pagina['Indicador']

    # DatosFormulario
    EpocasRed = Valores_pagina['EpocasRed']
    batch_size = Valores_pagina['batch_size']
    validation_split = Valores_pagina['validation_split']
    early_stopping = Valores_pagina['early_stopping']

    Poblacion = Valores_pagina['Poblacion']
    Generaciones = Valores_pagina['Generaciones']
    ProbCruce = Valores_pagina['ProbCruce']
    ProbMutacion = Valores_pagina['ProbMutacion']

    mu= Valores_pagina['Gama'] 
    sigma= Valores_pagina['Sigma']
    # indpb=Valores_pagina['Poblacion']
    
    # VALIDAR PARAMETROS ANTES DE ENTRENAR RED
    if EpocasRed == "":EpocasRed = 3
    if batch_size == "":batch_size = 32
    if validation_split == "":validation_split = 0.2
    if early_stopping == "":early_stopping =1

    # VALIDAR PARAMETROS DEL ALGORTIMO
    if Poblacion == "":Poblacion = 10
    if Generaciones == "":Generaciones = 5
    if ProbCruce == "":ProbCruce = 0.55
    if ProbMutacion == "":ProbMutacion =0.22
    if mu == "":mu =0.22
    if sigma == "":sigma =0.22

    #IMPRITMIR PARA METROS DE ENTRADA
    print("Datos Red Neuronal")
    print(f"{EpocasRed} + {batch_size} + {validation_split} + {early_stopping}")
    print("DatosAlgortimo")
    print(f"{Poblacion} + {Generaciones} + {ProbCruce} + {ProbMutacion}")

    # EJECUTAR MODELO O MODELO CON ALGORTIMO
    if Indicador == "1":
        print("Solo red")
        modeloRed(int(EpocasRed),int(batch_size),int(early_stopping),int(validation_split))
    elif Indicador == "2":
        print("Algortimo")
        # modeloAlgoritmo()
    else:
        print("Opcion no valida por favor validar")
    return render_template('Entrenamiento.html')

#Cargar el modelo
def Cargarmodelo():   
    global  loaded_model
    loaded_model = load_model('Proyecto de Grado/modelo.h5')
    DatosModelo = loaded_model.summary()
    # Verificar la arquitectura actual del modelo
    print(DatosModelo)
    # Capturar la salida de summary()
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        loaded_model.summary()
        summary_str = buf.getvalue()


# Evaluar el modelo
@Pagina.route('/EvaluarModelo')
def Evaluarmodelo(loaded_model,x_test,y_test):  
    # CARAGR MODELO
    print("Cragando Modelo")
    Cargarmodelo()
    print("Modelo Cargado")

    # EVALUAR MODELO    
    loss, accuracy = loaded_model.evaluate(x_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')


#Dirección principal
@Pagina.route('/')
def index():
    Valores_pagina = {'nombreJugador1': 'Juan', 'nombreJugador2': 'Pedro'}    
    juego=[0,0]  
    return render_template('SistemaMonitoreo.html',Var=Valores_pagina,Variable=juego)

# -----------------------------------------------------------------------------------------------------
# REDIRIGIR A OTRAS PAGINAS
@Pagina.route('/SistemaMonitoreo')
def Respuesta():  
        Valores_pagina = {'nombreJugador1': 'Juan', 'nombreJugador2': 'Pedro'}    
        juego=[0,0]              
        return render_template('SistemaMonitoreo.html',Var=Valores_pagina,Variable=juego)

@Pagina.route('/ReduccionDimensionalidad', methods = ['GET', 'POST'])
def ReduccionDimensioanlidad():    
    return render_template('ReduccionDimensionalidad.html')

@Pagina.route('/Entrenamiento', methods = ['GET', 'POST'])
def Entrenamiento():    
    return render_template('Entrenamiento.html')

@Pagina.route('/EjecutarModeloDato', methods = ['GET', 'POST'])
def EjecutarModeloFront():        
    return render_template('EjecutarModelo.html')

# -----------------------------------------------------------------------------------------------------
@Pagina.route('/PDF')
def PDF():      
    print("Armando PDF")   
    # BD = mysql.connect(host=ORIGEN, user=USUARIO, passwd=CONTRASENA, db=BASEDATOS)
    # Cursor = BD.cursor()    
    # Cursor.execute("SELECT * FROM MONITOREO")
    # results=Cursor.fetchall()
    # df = pd.DataFrame(results, columns=Cursor.column_names)
    # BD.commit()
    # """Cerrar la BD"""
    # BD.close()    
    
    # print(df)
    # pdf = FPDF()            
    # pdf.add_page(orientation='L')   
    # pdf.set_font('Times', '', 12)        
    # # Títulos de las columnas
    # for col in df.columns:
    #     pdf.cell(23, 10, col, border=1)
    # pdf.ln()

    # Datos de la tabla
    # for i in range(df.shape[0]):
    #     for j in range(df.shape[1]):                                      
    #         pdf.cell(23, 15, str(df.iloc[i, j]) [0:9], border=1)
    #     pdf.ln()

    # #Guardar pdf
    # pdf.output('ProyectoCorte2Monitoreo/data.pdf', 'F')     
    return render_template('PaginaMonitoreo.html')


# ACTUALIZAR PAGINAS
@Pagina.route('/ActualizarInformacion')
def ActualizarInformacion():
    # CONSULTAR INFROMACION DE LOS PCA 
    BD = mysql.connect(host=ORIGEN, user=USUARIO, passwd=CONTRASENA, db=BASEDATOS)
    Cursor = BD.cursor()       
    Cursor.execute("SELECT * FROM ReduccionPCA")
    MSG = MIMEMultipart()
    html_table=""
    for row in Cursor:
        html_table += "<tr>"
        for value in row:
            html_table += "<td>{}</td>".format(value)
        html_table += "</tr>"

    #CERRAR CONEXION 
    BD.commit()               
    BD.close()
    return jsonify({'TablaPCA': html_table})

@Pagina.route('/ActualizarInformacionEntrenamiento')
def ActualizarInformacionEntrenamiento():
    # CONSULTAR INFROMACION DE LOS PCA 
    BD = mysql.connect(host=ORIGEN, user=USUARIO, passwd=CONTRASENA, db=BASEDATOS)
    Cursor = BD.cursor()       
    Cursor.execute("SELECT * FROM RedNeuronal")    
    html_table=""
    for row in Cursor:
        html_table += "<tr>"
        cont=0
        for value in row:
            cont +=1
            if cont>2:
                value = value[:4]
            html_table += "<td>{}</td>".format(value)
        html_table += "</tr>"

    #CERRAR CONEXION 
    BD.commit()               
    BD.close()
    return jsonify({'TablaRed': html_table})


'''Función principal'''
if __name__ == "__main__":    
    #CONCEXION BASE DE DATOS 
    """Crear variables con los parámetros de acceso a la BD"""
    ORIGEN="localhost"
    USUARIO="root"
    CONTRASENA="1234"
    BASEDATOS="universidad"

    # PaginaWEB
    Pagina.run()
     
