#!/usr/bin/env python3
import requests

def main():
    # URL del endpoint de la API
    url = "http://localhost:5000/predict"
    
    # Construir imágenes de ejemplo:
    # Se crean dos imágenes de 28x28 píxeles con 3 canales cada una.
    # En este ejemplo, cada píxel se define con el valor 1.0 (imágenes normalizadas).
    input_ref = [[[1.0, 1.0, 1.0] for _ in range(28)] for _ in range(28)]
    input_query = [[[1.0, 1.0, 1.0] for _ in range(28)] for _ in range(28)]
    
    # Característica adicional: por ejemplo, 0.5
    input_carac = [0.5]
    
    # Crear el payload JSON con la estructura que espera la API
    payload = {
        "input_ref": input_ref,
        "input_query": input_query,
        "input_carac": input_carac
    }
    
    print("Enviando petición a la API en:", url)
    
    try:
        response = requests.post(url, json=payload)
    except Exception as e:
        print("Error al conectar con la API:", e)
        return
    
    if response.status_code == 200:
        print("Respuesta de la API:")
        print(response.json())
    else:
        print("Error en la petición:")
        print("Código de estado:", response.status_code)
        print("Detalle:", response.text)

if __name__ == '__main__':
    main()
