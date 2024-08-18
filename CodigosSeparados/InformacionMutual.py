import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# Leer el archivo Excel
df = pd.read_excel('Proyecto de Grado/DataSetPruebas.xlsx')

# Suponiendo que la última columna es la etiqueta y el resto son características
X = df.iloc[:, :-1]  # Todas las columnas excepto la última
y = df.iloc[:, -1]   # La última columna

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
plt.savefig('Proyecto de Grado\static\Imagenes\ImportaciaCaractDatos.png', bbox_inches='tight')
plt.show()
# Guardar los resultados en un archivo CSV
# mi_df.to_csv('mutual_information_results.csv')
print("Información mutua calculada y guardada en 'mutual_information_results.csv'.")
