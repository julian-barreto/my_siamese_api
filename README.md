# my_siamese_api
Modelo deep learning para optimizar los costos asociados al fraude tras analizar rutas de paquetes expuestas a algun tipo de fraude

# Siamese API con Flask y TensorFlow

Este repositorio contiene un ejemplo de aplicación Python que integra:

- **Una API REST con Flask:** Sirve un modelo de red neuronal siamesa para realizar predicciones.
- **Un modelo de red neuronal siamesa en TensorFlow/Keras:** Entrenado para resolver un problema de clasificación asociados al fraude utilizando imágenes simuladas.
- **Entrenamiento y validación:** Incluye un script que genera datos simulados, divide los datos en conjuntos de entrenamiento y validación, y entrena el modelo utilizando una función de pérdida personalizada con costos dinámicos.
- **Docker:** Se incluye un `Dockerfile` para construir una imagen y ejecutar la API en un contenedor.

## Estructura general del repositorio

```
my_siamese_api/
├── app.py           # API de Flask que carga el modelo y expone el endpoint /predict.
├── model.py         # Definición de la arquitectura de la red neuronal siamesa y funciones personalizadas.
├── train.py         # Script para generar datos simulados, entrenar el modelo y guardarlo en formato H5.
├── requirements.txt # Dependencias requeridas (Flask, TensorFlow, NumPy).
└── Dockerfile       # Instrucciones para construir la imagen Docker.
```

## Requisitos

- **Python**
- **TensorFlow**
- **Flask**
- **NumPy**
- **Matplotlib**
- **Pandas**

Se recomienda utilizar un entorno virtual para instalar las dependencias.

## Instalación

1. **Clona el repositorio:**

   ```bash
   git clone #url
   cd my_siamese_api
   ```

2. **Crea y activa un entorno virtual (opcional, pero recomendado):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # En Linux/Mac
   venv\Scripts\activate      # En Windows
   ```

3. **Instala las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

## Entrenamiento del modelo

El script `train.py` se encarga de:

- **Generar datos simulados:**
  - Crea 400 imágenes de 28x28 píxeles con 3 canales (RGB), donde la mayoría son blancas y en una fila aleatoria se asignan valores de color aleatorios.
- **Formar pares de imágenes:**
  - Divide las imágenes en dos grupos imagen de referencia (como debia ser una ruta) e imagen de consulta (como se esta ejecutando la ruta) y asigna a cada par:
    - Una característica adicional (valor entre 0 y 1). (puede ser una caracteristica de los paquetes o el paquete que transporta)
    - Un costo (valor aleatorio entre 10 y 100). 
    - Una etiqueta binaria (1 para ruta sin fraude, 0 para ruta con fraude).
- **División en entrenamiento y validación:**
  - Separa el 80% de los datos para entrenamiento y el 20% para validación.
- **Entrenamiento con función de pérdida personalizada:**
  - Utiliza una función de pérdida asimétrica que incorpora el costo de cada muestra (dinámico por batch): 
  Por cada transacción el porcentaje de ganancia es de un 25%, y por cada fraude aprobado se pierde el 100% del dinero de la transacción.
- **Guardado del modelo:**
  - Guarda el modelo entrenado en el archivo `model.h5`, que es el que utiliza la API para realizar predicciones.

Para entrenar el modelo, ejecuta:

```bash
python train.py
```

## API REST

El archivo `app.py` implementa una API con Flask que carga el modelo `model.h5` y expone el endpoint `/predict`.

### Endpoint `/predict`

- **Método:** POST  
- **Descripción:** Recibe datos en formato JSON para realizar una predicción.
- **Datos esperados en el JSON:**
  - `input_ref`: Imagen ruta de referencia (lista anidada que representa un array de 28x28x3, con valores normalizados).
  - `input_query`: Imagen ruta de consulta (misma estructura que `input_ref`).
  - `input_carac`: Característica adicional (por ejemplo, `[0.5]`).


## Ejecución de la API

Para iniciar la API de Flask, asegúrate de que el modelo `model.h5` exista (entrena el modelo previamente con `train.py`) y luego ejecuta:

```bash
python app.py
```

La API se ejecutará en `http://localhost:5000`.

### Ejemplo de petición usando Python

El archivo `peticion.py` contiene un ejemplo ejecutable con una peticion al api.

## Uso con Docker

El repositorio incluye un `Dockerfile` que permite construir una imagen Docker para ejecutar la API en un contenedor.

1. **Construir la imagen Docker:**

   ```bash
   docker build -t my_siamese_api .
   ```

2. **Ejecutar el contenedor:**

   ```bash
   docker run -p 5000:5000 my_siamese_api
   ```

La API estará disponible en `http://localhost:5000`.

## Notas adicionales

- **Antes de ejecutar la API (`app.py`), asegúrate de haber entrenado el modelo ejecutando `train.py`, ya que el modelo se carga desde el archivo `model.h5`.**

- **La pérdida asimétrica utiliza el costo de cada muestra de forma dinámica. Durante el entrenamiento, se espera que la variable de salida `y_true` tenga dos columnas: la etiqueta y el costo de la transacción.**



# CARPETA ADICIONAl: data_transform

## Transformación de Datos a Imágenes

Este carpeta contiene un script que convierte un conjunto de datos con información sobre el recorrido en una ruta de paquetes en una ciudad en imágenes de tamaño **28x28x3**.

## Estructura de la Imagen
- **Canal Rojo (R):** Representa las coordenadas normalizadas.
- **Canal Verde (G):** Representa el tiempo de llegada al punto.
- **Canal Azul (B):** Representa la categoría del paquete (1, 2 o 3).

## Requisitos
Antes de ejecutar el script, asegúrese de tener instaladas las siguientes librerías:

```bash
pip install numpy pandas matplotlib pillow
```

## Uso
Ejecute el script `data_transform.py` para generar la imagen:

```bash
python data_transform.py
```

Esto generará una imagen **"recorrido_paquete_fixed.png"** en la misma carpeta del script.

## Salida
La imagen resultante es una representación de los datos del recorrido de los paquetes en la ciudad, donde la intensidad de los colores refleja la información relevante.

# CARPETA ADICIONAl: Generative Model 

## Descripción de la Carpeta `generative_model`
Esta carpeta contiene un modelo sugerido y no ejecutable con una Generative Adversarial Network (GAN) que se utiliza para aumentar la cantidad de imágenes disponibles en el conjunto de datos de entrenamiento. Este modelo genera imágenes sintéticas similares a las del conjunto original, lo que permite mejorar la capacidad de generalización de una red neuronal entrenada con dichos datos.

## Ventajas de Aumentar Datos con un Modelo Generativo
El uso de una GAN para aumentar los datos de entrenamiento presenta múltiples ventajas:

1. **Mayor cantidad de datos**: Genera imágenes adicionales que pueden complementar el conjunto de entrenamiento.
2. **Mejor generalización**: Ayuda a la red neuronal a aprender representaciones más robustas y evitar sobreajuste.
3. **Equilibrio de clases**: Permite generar más muestras de clases poco representadas en los datos originales.
4. **Aumento de la diversidad**: Introduce variabilidad en las imágenes, lo que mejora la capacidad del modelo para adaptarse a datos del mundo real.
5. **Menos dependencia de la recolección manual de datos**: Reduce el esfuerzo y costo de obtener nuevas imágenes reales.

## Contenido de la Carpeta
- `generative.ipynb`: Implementación de un modelo generador de imágenes basado en GAN.
- `training_checkpoints/`: Carpeta donde se almacenan los checkpoints del entrenamiento.
- `image_epoch/`: Carpeta donde se guardan las imágenes generadas en cada época de entrenamiento.

## Uso del Modelo

El modelo generará nuevas imágenes y guardará los checkpoints del entrenamiento. Estas imágenes pueden utilizarse como datos de entrada para mejorar la precisión y robustez de otros modelos de visión por computadora.


## Autor
- **Julián Mauricio Rodríguez Barreto**
