# # -- coding: utf-8 --
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
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import icmp

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# CARGUE DE DATOS PREPROCESADOS
def load_dataset():    
    # Eliminar ataques innecesarios
    data = pd.read_excel('Proyecto de Grado\DataSedPCA_FINAL1.xlsx')
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
    data = data[~data['Tipo de Ataque'].isin(valores_a_eliminar)]
    data = data.drop(index=1)
    print("Guardando Dataset Filtrado")
    data.to_excel('Proyecto de Grado/DataSedPCA_FINAL21.xlsx', index=False)
    print("Documento Creado")

    # CARGAR CONJUNTO FINAL
    print("Leyendo Archivo")
    data = pd.read_excel('Proyecto de Grado\DataSedPCA_FINAL2.xlsx')

    inputs = data.iloc[:, :-1]
    Outputs = data.iloc[:, -1]

    encoder = OneHotEncoder(sparse=False)
    Outputs = encoder.fit_transform(Outputs.values.reshape(-1, 1))

    scaler = MinMaxScaler()
    inputs = scaler.fit_transform(inputs)
    x_train, x_test, y_train, y_test = train_test_split(inputs, Outputs, test_size=0.3, random_state=42)

    joblib.dump(encoder, 'Proyecto de Grado/encoder.joblib')
    joblib.dump(scaler, 'Proyecto de Grado/scaler.joblib')
    return x_train, x_test, y_train, y_test

# CARGAR DATOS DE DATASET
x_train, x_test, y_train, y_test = load_dataset()

# VARIABLES PARA EL MEJOR MODELO
best_model = None
best_accuracy = 0.0

# FUNCION OBJETIVO
def evaluate_nn(individual):
    neurons_layer1 = int(individual[0])
    neurons_layer2 = int(individual[1])
    learning_rate = individual[2]
    dropout_rate = individual[3]

    model = Sequential()
    model.add(Dense(neurons_layer1, activation='relu', input_dim=x_train.shape[1], kernel_regularizer=l2(0.001)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(neurons_layer2, activation='relu', kernel_regularizer=l2(0.001)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    optimizer = Nadam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy, model


# FUNCION PRINCIPAL
def main():
    global best_model, best_accuracy

    # Crear el problema de optimización
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", np.random.randint, 5, 201)
    toolbox.register("attr_float", np.random.uniform, 0.0001, 0.1)
    toolbox.register("attr_dropout", np.random.uniform, 0.0, 0.5)
    toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.attr_int, toolbox.attr_int, toolbox.attr_float, toolbox.attr_dropout), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # GUARDAR EL MEJOR MODELO DEL INDIVIDUO
    def evaluate_and_save_best(individual):
        global best_model, best_accuracy
        accuracy, model = evaluate_nn(individual)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
        return accuracy,

    toolbox.register("evaluate", evaluate_and_save_best)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    #Definir parámetros
    population = toolbox.population(n=10) #POBLACION
    ngen = 5                              #GENERACIONES
    cxpb = 0.55
    mutpb = 0.3

    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

    if best_model is not None:
        best_model.save('Proyecto de Grado/mejor_modelo.h5')
        print(f"Mejor modelo guardado con una precisión de {best_accuracy}")


# EJECUCION DE CODIGO
if __name__ == "__main__":
    main()



