import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package='Custom', name='absolute_difference')
def absolute_difference(tensors):
    """
    Calcula la diferencia absoluta entre dos tensores.
    Esta función está registrada para que Keras pueda serializarla y cargarla sin problemas.
    """
    return tf.abs(tensors[0] - tensors[1])

def create_base_network(input_shape):
    """
    Crea la red base para extraer características de las imágenes.
    """
    input_tensor = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model(input_tensor, x)

def create_siamese_network(image_shape=(28, 28, 3)):
    """
    Crea la red neuronal siamesa que recibe:
      - Dos imágenes (de referencia y de consulta).
      - Una variable numérica adicional que representa una característica.
    """
    # Definir los inputs
    input_ref = Input(shape=image_shape, name='input_ref')
    input_query = Input(shape=image_shape, name='input_query')
    input_carac = Input(shape=(1,), name='input_carac')
    
    # Red base compartida
    base_network = create_base_network(image_shape)
    
    # Extraer características
    feat_ref = base_network(input_ref)
    feat_query = base_network(input_query)
    
    # Usar la función registrada en la capa Lambda
    diff = layers.Lambda(
        absolute_difference,
        output_shape=lambda input_shape: input_shape[0]
    )([feat_ref, feat_query])
    
    # Concatenar la diferencia con la característica adicional
    x = layers.Concatenate()([diff, input_carac])
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=[input_ref, input_query, input_carac], outputs=output)
    return model
