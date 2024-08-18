import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import warnings

# Ignorar las advertencias de tensorflow
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Función para cargar tu dataset y dividirlo en características (X) y etiquetas (y)
def load_dataset(file_path):    
    data = pd.read_excel(file_path, skiprows=1)
    #ELIMINAR VALORES QUE NO ESTEN EN LSO CONJUNTOS DE PRUEBA
    valores_a_eliminar = [0, 1, 2]

    # Eliminar filas donde la columna 'A' tenga cualquiera de los valores especificados
    # data = data[~data['A'].isin(valores_a_eliminar)]


    inputs = data.iloc[:, :-1]  # Todas las columnas excepto la última
    Outputs = data.iloc[:, -1]   # Última columna (variable objetivo)
    x_train, x_test, y_train, y_test = train_test_split(inputs, Outputs, test_size=0.3, random_state=42)
    
    # Normalización de los datos
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
       
    # One-hot encoding para las etiquetas de salida con handle_unknown='ignore'
    # encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_test = encoder.transform(y_test.values.reshape(-1, 1))  
    
    # Guardar el OneHotEncoder y el escalador ajustado
    joblib.dump(encoder, 'encoder.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    return x_train, x_test, y_train, y_test

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

    # Cargar los datos de entrenamiento
    file_path = 'Proyecto de Grado\DataSedPCA.xlsx'
    x_train, x_test, y_train, y_test = load_dataset(file_path)

    # Crear la red neuronal
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))  # Usa softmax para clasificación multiclase
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    toolbox.register("mate", tools.cxBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3) #4 - 8 muy gra
    toolbox.register("evaluate", fitness)

    # Definir parámetros
    population_size = 200
    generations = 50
    cxpb = 0.7  # Probabilidad de cruce  0.55 o 0.66
    mutpb = 0.2  # Probabilidad de mutación 0.3

    # Crear población inicial
    population = toolbox.population(n=population_size)

    # Lista para guardar los valores de fitness
    fitness_values = []

    # Ejecutar algoritmo genético
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population[:] = toolbox.select(offspring, k=len(population))

        # Guardar el mejor valor de fitness de la generación actual
        best_ind = tools.selBest(population, k=1)[0]
        fitness_values.append(best_ind.fitness.values[0])
        print(f'Generación {gen}, Mejor Fitness: {best_ind.fitness.values[0]}')

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

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Evaluar en el conjunto de entrenamiento
    train_loss, train_accuracy = model.evaluate(x_train, y_train)
    print(f'Training Loss: {train_loss}, Training Accuracy: {train_accuracy}')

    # Guardar el modelo y el escalador
    model.save('modelo.h5')
    # np.save('classes.npy', encoder.categories_)
    # np.save('scaler.npy', scaler)

    # Guardar los valores de fitness
    np.save('fitness_values.npy', fitness_values)
