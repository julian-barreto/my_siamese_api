from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
# Importa el módulo que contiene la función registrada
from model import absolute_difference  # Esto asegura que esté disponible al deserializar

app = Flask(__name__)

# Cargar el modelo; al haber importado absolute_difference, Keras lo encontrará.
model = tf.keras.models.load_model("model.h5", compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        input_ref = np.array(data['input_ref'], dtype=np.float32)
        input_query = np.array(data['input_query'], dtype=np.float32)
        input_carac = np.array(data['input_carac'], dtype=np.float32)
    except KeyError as e:
        return jsonify({"error": f"Falta el campo {str(e)}"}), 400

    # Ajustar las dimensiones si es necesario
    if input_ref.ndim == 3:
        input_ref = np.expand_dims(input_ref, axis=0)
    if input_query.ndim == 3:
        input_query = np.expand_dims(input_query, axis=0)
    if input_carac.ndim == 1:
        input_carac = np.expand_dims(input_carac, axis=-1)
    elif input_carac.ndim == 0:
        input_carac = np.array([[input_carac]])

    prediction = model.predict({
        'input_ref': input_ref,
        'input_query': input_query,
        'input_carac': input_carac
    })
    
    return jsonify({"prediction": prediction.tolist()})

@app.route('/')
def home():
    return "API de clasificación con red neuronal siamesa."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
