import pandas as pd
import numpy as np
from deap import creator, base, tools, algorithms
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ---------------------RED EJEMPELO 1 -----------------------------------------------------------------
# Cargar los datos
iris = load_iris()
# Leer el archivo de Excel
data = pd.read_excel('Proyecto de Grado\DataSetPruebas.xlsx')
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Definir la función de aptitud
def fitness_function(individual):
    # Decodificar el individuo en los pesos de la red neuronal
    weights = np.array(individual).reshape(-1, len(X_train[0]))
    print(weights)
    # Crear la red neuronal con los pesos
    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42, weights=weights)
    # Entrenar la red neuronal
    clf.fit(X_train, y_train)
    # Calcular la precisión
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred),

# Configurar los parámetros del algoritmo genético
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(X_train[0])*10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Definir la población y ejecutar el algoritmo genético
population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=True)


