import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from model import create_siamese_network

# Función de pérdida personalizada que utiliza el costo de cada muestra de forma dinámica.
# Se asume que y_true tiene dos columnas: 
#   - La primera columna es la etiqueta (0 o 1).
#   - La segunda columna es el valor de la transacción (costo) para esa muestra.
def custom_loss(y_true, y_pred):
    # Extraer la etiqueta y el valor de la transacción de y_true
    label = y_true[:, 0]
    transaction_value = y_true[:, 1]
    gain_true = 0.25 * transaction_value  # Ganancia si la firma es auténtica
    loss_false = transaction_value        # Pérdida si la firma es falsa
    loss = - (label * gain_true * K.log(y_pred + 1e-9)) - ((1 - label) * loss_false * K.log(1 - y_pred + 1e-9))
    return K.mean(loss)

def create_images(num_images=400, height=28, width=28, channels=3):
    """
    Genera num_images imágenes de 28x28x3, todas blancas salvo por una fila
    aleatoria en la que se asignan colores aleatorios.
    """
    images = np.ones((num_images, height, width, channels), dtype=np.uint8) * 255
    for i in range(num_images):
        # Elegir una fila aleatoria
        row = np.random.randint(0, height)
        # Asignar colores aleatorios a la fila
        images[i, row, :, :] = np.random.randint(0, 256, (width, channels), dtype=np.uint8)
    # Normalizar imágenes a [0, 1]
    images = images.astype('float32') / 255.0
    return images

def generate_data():
    """
    Genera el dataset simulando:
      - 400 imágenes que se dividen en dos conjuntos (referencia y consulta).
      - Una variable 'característica' para la imagen de referencia.
      - Un 'costo' para cada par.
      - Una etiqueta binaria (1: imagen verdadera, 0: fraude).
    """
    images = create_images(num_images=400, height=28, width=28, channels=3)
    # Dividir en dos conjuntos iguales: 200 imágenes para referencia y 200 para consulta
    num_pairs = images.shape[0] // 2
    imgs_ref = images[:num_pairs]
    imgs_query = images[num_pairs:2*num_pairs]
    
    # Variable adicional: característica de la imagen (valor entre 0 y 1)
    caracteristicas = np.random.random((num_pairs, 1))
    # Costo variable por imagen (valor entre 10 y 100)
    costos = np.random.uniform(10, 100, size=(num_pairs,))
    # Etiqueta: 1 (imagen verdadera) o 0 (fraude)
    etiquetas = np.random.randint(0, 2, size=(num_pairs,))
    
    return imgs_ref, imgs_query, caracteristicas, costos, etiquetas

def main():
    # Generar los datos simulados
    imgs_ref, imgs_query, carac, costos, etiquetas = generate_data()
    num_pairs = imgs_ref.shape[0]
    
    # Mezclar y dividir en conjuntos de entrenamiento (80%) y validación (20%)
    indices = np.arange(num_pairs)
    np.random.shuffle(indices)
    split_index = int(0.8 * num_pairs)
    train_idx = indices[:split_index]
    val_idx = indices[split_index:]
    
    imgs_ref_train = imgs_ref[train_idx]
    imgs_query_train = imgs_query[train_idx]
    carac_train = carac[train_idx]
    costos_train = costos[train_idx]
    etiquetas_train = etiquetas[train_idx]
    
    imgs_ref_val = imgs_ref[val_idx]
    imgs_query_val = imgs_query[val_idx]
    carac_val = carac[val_idx]
    costos_val = costos[val_idx]
    etiquetas_val = etiquetas[val_idx]
    
    # Combinar la etiqueta y el costo en un solo vector para cada muestra.
    # La primera columna es la etiqueta, la segunda el transaction_value.
    y_train = np.column_stack([etiquetas_train, costos_train])
    y_val = np.column_stack([etiquetas_val, costos_val])
    
    # Crear y compilar el modelo utilizando la función de pérdida personalizada.
    model = create_siamese_network(image_shape=(28, 28, 3))
    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    
    # Entrenar el modelo utilizando también el conjunto de validación.
    model.fit(
        x = {
            'input_ref': imgs_ref_train,
            'input_query': imgs_query_train,
            'input_carac': carac_train
        },
        y = y_train,
        validation_data=(
            {
                'input_ref': imgs_ref_val,
                'input_query': imgs_query_val,
                'input_carac': carac_val
            },
            y_val
        ),
        epochs = 10,
        batch_size = 16
    )
    
    # Guardar el modelo entrenado en formato H5.
    model.save("model.h5")
    print("Modelo guardado en 'model.h5'.")

if __name__ == '__main__':
    main()
