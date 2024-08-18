# import tensorflow as tf
# from tensorflow.keras import layers, models

# # ---------------------RED EJEMPELO 1 -----------------------------------------------------------------
# # Definir la arquitectura de la red neuronal
# model = models.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(10,)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# # Compilar el modelo
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Datos de ejemplo
# import numpy as np
# x_train = np.random.rand(1000, 10)
# y_train = np.random.randint(0, 2, (1000,))

# print(x_train)

# # Entrenar el modelo
# model.fit(x_train, y_train, epochs=5, batch_size=32)

# # Modificar el peso de la primera capa oculta en la posición (0, 0)
# new_weight = 0.4  # Nuevo valor del peso
# weights = model.get_weights()
# weights[0][0][0] = new_weight  # Modificar el peso
# # print(weights)
# model.set_weights(weights)  # Establecer los nuevos pesos

# # Evaluar el modelo con los pesos modificados
# loss, accuracy = model.evaluate(x_train, y_train)
# print(f'Loss: {loss}, Accuracy: {accuracy}')
