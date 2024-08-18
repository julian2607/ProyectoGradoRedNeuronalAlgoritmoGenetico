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
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# Ignorar las advertencias de tensorflow
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Función para cargar tu dataset y dividirlo en características (X) y etiquetas (y)
def load_dataset():    
    # data = pd.read_excel('Proyecto de Grado\DataSedPCA.xlsx', skiprows=1)
    data = pd.read_excel('Proyecto de Grado\DataSedPCA.xlsx')

    # Valores a eliminar
    valores_a_eliminar = [
        'Uploading_attack','Recon-PingSweep','XSS',
        'Backdoor_Malware','CommandInjection','SqlInjection',
        'BrowserHijacking','DictionaryBruteForce',
        'DDoS-SlowLoris','DDoS-HTTP_Flood',
        'VulnerabiliFyScan','DoS-HTTP_Flood',
        'Recon-OSScan','Recon-HostDiscovery',
        'DNS_Spoofing','DDoS-UDP_Fragmentation',
        'DDOS_ASK_Frafmetation','MITM-ArpSpoofing','DDoS-ICMP_Fragmentation'
        'Mirai-greip_flood','DoS-HTTP_Flood','DDoS-ICMP_Fragmentation',
        'BenignTraffic','MITM-ArpSpoofing','Mirai-udpplain',
        'Mirai-greeth_flood','DoS-SYN_Flood','DoS-TCP_Flood',
        'VulnerabilityScan','BenignTraffic','Uploading_Attack',
        'Recon-PortScan','Mirai-udpplain','Mirai-greip_flood',
        'Mirai-greeth_flood','DoS-UDP_Flood','DoS-TCP_Flood',
        'DoS-SYN_Flood','DDoS-ACK_Fragmentation'
    ]

    # Eliminar filas donde la columna 'A' tenga cualquiera de los valores especificados
    data = data[~data['Tipo de Ataque'].isin(valores_a_eliminar)]
    # Eliminar la primera fila
    data = data.drop(index=1)
    data.to_excel('Proyecto de Grado/DataSedPCA_FINAL2.xlsx', index=False)


    #LEER NUEVAMENTE EL DATASET YA FILTRADO
    data = pd.read_excel('Proyecto de Grado\DataSedPCA_FINAL2.xlsx')
    
    inputs = data.iloc[:, :-1]  # Todas las columnas excepto la última
    Outputs = data.iloc[:, -1]   # Última columna (variable objetivo)

    # Codificar las etiquetas de salida usando One-Hot Encoding
    encoder = OneHotEncoder(sparse=False)
    Outputs = encoder.fit_transform(Outputs.values.reshape(-1, 1))

    # Normalizar las columnas de entrada
    scaler = MinMaxScaler()
    inputs = scaler.fit_transform(inputs)
    x_train, x_test, y_train, y_test = train_test_split(inputs, Outputs, test_size=0.3, random_state=42)
    
    # One-hot encoding para las etiquetas de salida con handle_unknown='ignore'
    # encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) 
    
    # Guardar el OneHotEncoder ajustado
    joblib.dump(encoder, 'Proyecto de Grado/encoder.joblib')
    joblib.dump(scaler, 'Proyecto de Grado/scaler.joblib')
    return x_train, x_test, y_train, y_test

# Función para crear la red neuronal
def create_model(weights):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=x_train.shape[1], kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))  # Usa sigmoid para clasificación multiclase
    model.set_weights(weights)
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Evaluar el modelo
def evaluate_model(individual):
    weights = decode_weights(individual, model)
    model.set_weights(weights)
    loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
    return accuracy,

# Decodificar los pesos desde el cromosoma
def decode_weights(individual, model):
    shapes = [w.shape for w in model.get_weights()]
    weights = []
    start = 0
    for shape in shapes:
        size = np.prod(shape)
        weights.append(np.array(individual[start:start + size]).reshape(shape))
        start += size
    return weights

# Crear la red neuronal con pesos iniciales
x_train, x_test, y_train, y_test = load_dataset()
model = create_model(model.get_weights())

# Configurar el algoritmo genético
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(np.concatenate([w.flatten() for w in model.get_weights()])))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_model)

# Ejecutar el algoritmo genético
def main():
    random.seed(42)
    population = toolbox.population(n=20)
    ngen, cxpb, mutpb = 10, 0.5, 0.2
    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population[:] = offspring
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    best_weights = decode_weights(best_ind, model)
    model.set_weights(best_weights)
    model.save('Proyecto de Grado/modelo_mejorado.h5')

if __name__ == "__main__":
    main()
