import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Función para cargar tu dataset y dividirlo en características (X) y etiquetas (y)
def load_dataset():
    global inputs,Outputs
    data = pd.read_excel('Proyecto de Grado\DataSedPCA.xlsx')
    inputs = data.iloc[:, :-1]  # Todas las columnas excepto la última
    Outputs = data.iloc[:, -1]   # Última columna (variable objetivo)
    return inputs, Outputs

# Funcion fitness para calcualr el mejor objetivo
def fitness_function(individual, inputs, Outputs):
    correct = 0
    for i in range(len(inputs)):
        prediction = sum(x * w for x, w in zip(inputs.iloc[i], individual))
        predicted_attack_type = "DDoS-RSTFINFlood" 
        # Comparar el tipo de ataque predicho con el real
        if predicted_attack_type == Outputs.iloc[i]: 
            correct += 1            
    return correct / len(inputs),

# ------------------------------------------------------------------------------------------------
# Función para predecir la salida dada una entrada utilizando el mejor individuo encontrado
def predict(input_data, best_individual):
    print("Predecir entrada")
    prediction = sum(x * w for x, w in zip(input_data, best_individual))
    return prediction

# ------------------------------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
if __name__ == "__main__":
    # Cargar informacion
    inputs,Outputs = load_dataset()
    # Configuración de DEAP
    creator.create("FitnessMax", base.Fitness, weights=(4.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 5, 10)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(inputs.columns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_function, inputs=inputs, Outputs=Outputs)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Configuración de la población
    population_size = 1
    num_generations = 1
    population = toolbox.population(n=population_size)

    # Ejecutar el algoritmo genético
    for gen in range(num_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    # Obtener el mejor individuo
    best_ind = tools.selBest(population, k=1)[0]
    accuracy = fitness_function(best_ind, inputs, Outputs)[0]
    print(f"El mejor individuo es: {best_ind}, con un fitness de: {accuracy}")

    # Usar el mejor individuo encontrado para predecir la salida para una nueva entrada
    # new_input = (3, 4)  # Nueva entrada
    # best_individual = tools.selBest(population, k=1)[0]  # Mejor individuo encontrado
    # prediction = predict(new_input, best_individual)
    # print(f"Para la entrada {new_input}, la predicción es: {prediction}")



    # ------------------------ EJEMPLO CON COLAB -------------------------------------------#
    # import tensorflow as tf
    # import numpy as np
    # celsius=np.array([-40,-10,0,8,15,22,38],dtype=float)
    # fahrentheit = np.array([-40,14,32,46,59,72,100],dtype=float)
    # capa = tf.keras.layers.Dense(units=1,input_shape=[1])
    # modelo = tf.keras.Sequential([capa])
    # modelo.compile(
    # optimizer=tf.keras.optimizers.Adam(0.1),
    # loss = 'mean_squared_error'
    # )
    # print("Comenzar entrenamiento....")
    # historial = modelo.fit(celsius,fahrentheit,epochs=1000,verbose=False)
    # print("Modelo Entrenado")

    # ---------------------------- EJEMPLO 2 -----------------------------------------------#
    # import tensorflow as tf
    # from tensorflow.keras.datasets import mnist
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Dense, Flatten
    # # Cargar y preprocesar el conjunto de datos MNIST
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # # Construir el modelo
    # model = Sequential([
    #     Flatten(input_shape=(28, 28)),
    #     Dense(128, activation='relu'),
    #     Dense(10, activation='softmax')
    # ])
    # # Compilar el model
    # model.compile(optimizer='adam',
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])
    # # Entrenar el modelo
    # model.fit(x_train, y_train, epochs=5)
    # # Evaluar el modelo
    # model.evaluate(x_test, y_test)

