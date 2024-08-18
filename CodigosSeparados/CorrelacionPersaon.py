import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Carga el archivo Excel
df = pd.read_excel('Proyecto de Grado/DataSetPruebas.xlsx')

# Aplicar one-hot encoding a las columnas categóricas
df_encoded = pd.get_dummies(df, columns=['label'])

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
plt.savefig('Proyecto de Grado\static\Imagenes\MatrizCorrelacion2.png', bbox_inches='tight')
plt.show()


