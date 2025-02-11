import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Parámetros
image_size = (28, 28)  # Tamaño de la imagen
num_points = 100  # Número de puntos en el recorrido

# Generar datos de coordenadas aleatorias en una ciudad (valores normalizados entre 0 y 1)
np.random.seed(42)
x_coords = np.random.rand(num_points)
y_coords = np.random.rand(num_points)

# Generar tiempos de llegada aleatorios (valores entre 0 y 100)
time_arrival = np.random.randint(0, 100, num_points)

# Generar categorías de paquetes entre [1, 2, 3]
categories = np.random.choice([1, 2, 3], num_points)

# Crear dataframe
df = pd.DataFrame({
    'x': x_coords,
    'y': y_coords,
    'time': time_arrival,
    'category': categories
})

# Asegurar que los valores sean enteros
df['x_pixel'] = df['x'].astype(int)
df['y_pixel'] = df['y'].astype(int)

# Crear una nueva imagen vacía
image_fixed = np.zeros((28, 28, 3), dtype=np.uint8)

# Asignar los valores corregidos a la imagen
for index, row in df.iterrows():
    x_pixel = int(row['x_pixel'])
    y_pixel = int(row['y_pixel'])
    image_fixed[x_pixel, y_pixel, 0] = int(row['x'])  # R: Coordenada X
    image_fixed[x_pixel, y_pixel, 1] = int(row['time'])  # G: Tiempo de llegada
    image_fixed[x_pixel, y_pixel, 2] = int(row['category'])  # B: Categoría del paquete

# Convertir la imagen en formato PIL y guardarla
img_fixed = Image.fromarray(image_fixed)
image_fixed_path = "recorrido_paquete_fixed.png"
img_fixed.save(image_fixed_path)


# Confirmación de guardado
image_fixed_path
