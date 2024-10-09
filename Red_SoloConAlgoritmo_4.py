import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import joblib
import warnings
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib
import warnings
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Cargar el dataset
def load_dataset():
    data = pd.read_excel('Proyecto de Grado\DataSedPCA_FINAL2 _PRUEBA.xlsx')
    inputs = data.iloc[:, :-1]
    Outputs = data.iloc[:, -1]
    encoder = OneHotEncoder(sparse=False)
    Outputs = encoder.fit_transform(Outputs.values.reshape(-1, 1))
    scaler = MinMaxScaler()
    inputs = scaler.fit_transform(inputs)
    x_train, x_test, y_train, y_test = train_test_split(inputs, Outputs, test_size=0.3, random_state=42)
    joblib.dump(encoder, 'Proyecto de Grado\DatosProcesados\encoder.joblib')
    joblib.dump(scaler, 'Proyecto de Grado\DatosProcesados\scaler.joblib')
  
  # Guardar los datos en archivos .npy
    np.save('Proyecto de Grado/DatosProcesados/x_train.npy', x_train)
    np.save('Proyecto de Grado/DatosProcesados/x_test.npy', x_test)
    np.save('Proyecto de Grado/DatosProcesados/y_train.npy', y_train)
    np.save('Proyecto de Grado/DatosProcesados/y_test.npy', y_test)
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_dataset()

def create_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=x_train.shape[1], kernel_regularizer=l2(0.001)))    
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))    
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    return model

model = create_model()

# Obtener todos los pesos de la red y aplanarlos en un vector
def get_weights(model):
    weights = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        weights.extend(layer_weights[0].flatten())
        weights.extend(layer_weights[1].flatten())
    return np.array(weights)

# Establecer los pesos de la red desde un vector
def set_weights(model, weights):
    new_weights = []
    index = 0
    for layer in model.layers:
        layer_weights = layer.get_weights()
        weight_shape = layer_weights[0].shape
        bias_shape = layer_weights[1].shape
        weight_size = np.prod(weight_shape)
        bias_size = np.prod(bias_shape)
        new_weights.append(weights[index:index + weight_size].reshape(weight_shape))
        index += weight_size
        new_weights.append(weights[index:index + bias_size].reshape(bias_shape))
        index += bias_size
        layer.set_weights(new_weights[-2:])

# FUNCION 1 PRUEBAS
# def evaluate_nn(weights):
#     set_weights(model, weights)
#     optimizer = Nadam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy','Precision','recall'])
#     early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#     history = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
#     _, accuracy,precision,recall = model.evaluate(x_test, y_test, verbose=0)
#     return (accuracy,precision,recall,)

# FUNCION 2 
def evaluate_nn(weights):
    set_weights(model, weights)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test,verbose=0)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
    print("acurracy: %s  precision: %s recall: %s"%(accuracy,precision,recall) )
    return (accuracy,precision,recall,)  

creator.create("FitnessMax", base.Fitness, weights=(1.0,3.5,0.5))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -1, 1, len(get_weights(model)))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_nn)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Función similar para el Hall of Fame
def similar(ind1, ind2):
    return np.allclose(ind1, ind2, atol=1e-6)

def main():
    #PARAMETROS DEL ALGORTIMO 
    population = toolbox.population(n=10) #Poblacion
    ngen = 3 #Generaciones
    cxpb = 0.55 #Probabilidad de cruce
    mutpb = 0.22 #Probabilidad de mutación

    #ESTADISITCAS DE POBLACION ENTRENAMIENTO
    hof = tools.HallOfFame(1, similar=similar)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # ESTABLECES ALGORTIMO
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, halloffame=hof, verbose=True)

    #MEJOR INDIVIDUO
    print("---------------------------------------------------------------------------")
    best_individual = hof[0]
    best_weights = np.array(best_individual)
    set_weights(model, best_weights)
    # Imprime el mejor individuo absoluto guardado en hof
    print("El mejor individuo es: %s with fitness: %s" % (best_individual, best_individual.fitness.values))

    # MEJOR AL FINAL DEL PROCESO
    print("---------------------------------------------------------------------------")
    best_individual2 = tools.selBest(population, 1)[0]
    accuracy,presicion,recall = best_individual2.fitness.values
    # print("loss: %s and accuracy:" % (accuracy))
    print("Best individual is: %s\nwith fitness: %s" % (best_individual, best_individual.fitness.values))

    # Aplicar los pesos del mejor individuo al modelo
    print("---------------------------------------------------------------------------")
    print("Colocando pesos")
    set_weights(model, np.array(best_individual2))
    evaluate_nn(np.array(best_individual2))

    # Evaluar el modelo en el conjunto de prueba y obtenere medidas para medir el modelo
    y_pred = model.predict(x_test)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
    print(f'Exactitud: {accuracy}')
    print(f'Precisión: {precision}')
    print(f'Sensibilidad: {recall}')

    #Guardar el modelo
    model.save('Proyecto de Grado/mejor_modelo_pesos.h5')
    print("Mejor modelo guardado.")

if __name__ == "__main__":
    main()
