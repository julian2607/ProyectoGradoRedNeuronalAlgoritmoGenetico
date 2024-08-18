# -- coding: utf-8 --
# pylint: disable=import-error
import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings

# Ignorara las advertencia de tensroflow
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)

# Función para cargar tu dataset y dividirlo en características (X) y etiquetas (y)
def load_dataset():    
    data = pd.read_excel('Proyecto de Grado\DataSedPCA.xlsx', skiprows=1)
    inputs = data.iloc[:, :-1]  # Todas las columnas excepto la última
    Outputs = data.iloc[:, -1]   # Última columna (variable objetivo)
    x_train, x_test, y_train, y_test = train_test_split(inputs, Outputs, test_size=0.3, random_state=42)
    # Obtener todas las clases presentes en la salida
    all_classes = np.unique(Outputs)
    # One-hot encoding para las etiquetas de salida
    encoder = OneHotEncoder(categories=[all_classes.tolist()], sparse=False)
    y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_test = encoder.transform(y_test.values.reshape(-1, 1))  
    # Obtener todas las clases presentes en la salida
    all_classes = np.unique(Outputs)
    # Guardar el OneHotEncoder ajustado
    joblib.dump(encoder, 'Proyecto de Grado/encoder.joblib')
    return x_train, x_test, y_train, y_test,all_classes,encoder


# Función de fitness
def fitness(weights):
    # Obtener la estructura de los pesos de la red neuronal
    weights_structure = [w.shape for w in model.get_weights()]
        
    # Crear una lista de pesos con la estructura correcta
    new_weights = []
    start = 0
    for shape in weights_structure:
        size = np.prod(shape)
        new_weights.append(weights[start:start+size].reshape(shape))
        start += size

    # Establecer los nuevos pesos en la red neuronal    
    model.set_weights(new_weights)

    # Evaluar el modelo
    loss, _ = model.evaluate(x_train, y_train, verbose=0)
    return (-1.0 * loss,)  # Queremos maximizar el fitness, por eso multiplicamos por -1


# ------------------------------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
if __name__ == "__main__":   

    # Datos de entrenamiento DataSet    
    x_train, x_test, y_train, y_test,all_classes,encoder = load_dataset()

    # Crear la red neuronal
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))    
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # FUNCIONES DE ACTIVACION:
    # sigmoid,softmax

    # Definir la cantidad total de parámetros de la red neuronal
    total_params = sum([np.prod(w.shape) for w in model.get_weights()])

    # Crear el problema de optimización
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=total_params)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Registrar operadores genéticos
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness)

    # Definir parámetros
    population_size = 150
    generations = 5

    # Crear población inicial
    population = toolbox.population(n=population_size)
    
    # Ejecutar algoritmo genético
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population[:] = toolbox.select(offspring, k=len(population))

    # Obtener el mejor individuo
    best_ind = tools.selBest(population, k=1)[0]
    best_weights = np.array(best_ind)

    # Aplicar los mejores pesos a la red neuronal
    weights_structure = [w.shape for w in model.get_weights()]
    new_weights = []
    start = 0
    for shape in weights_structure:
        size = np.prod(shape)
        new_weights.append(best_weights[start:start+size].reshape(shape))
        start += size
    model.set_weights(new_weights)


    #Mirar caracteriticas de datos de entremamiento y 
    # las entradas que permite el modelo
    print(x_test.shape)
    print(model.input_shape)

    # Evaluar
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Guardar el modelo y Encoding
    model.save('Proyecto de Grado/modelo.h5')
    np.save('Proyecto de Grado/classes.npy', all_classes)
    np.save('Proyecto de Grado/encoder_categories.npy', encoder.categories_, allow_pickle=True)