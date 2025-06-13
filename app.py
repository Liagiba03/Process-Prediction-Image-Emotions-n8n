from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import cv2
import requests
from PIL import Image
import os
import io
import base64
import json

app = Flask(__name__)

# Diccionario para mapear etiquetas a emociones
label_to_text = {0: 'Ira', 1: 'Odio', 2: 'Miedo', 3: 'Felicidad', 4: 'Tristeza', 5: 'Sorpresa', 6: 'Neutral'}

@app.route('/processImage', methods=['POST'])
def preprocess_image():
    try:
        # Obtener la imagen desde la solicitud
        file = request.files.get('image')
        if not file:
            return jsonify({"error": "No se proporcionó ninguna imagen"}), 400

        # Leer la imagen y convertirla a escala de grises
        img = Image.open(file).convert('L')
        # Redimensionar la imagen al tamaño objetivo (por ejemplo, 48x48)
        target_size = (48, 48)
        img = img.resize(target_size)
        # Convertir la imagen a un array de NumPy
        img_array = np.array(img, dtype=np.float32)
        # Normalizar los valores de los píxeles al rango [0, 1]
        img_array = img_array / 255.0
        # Expandir dimensiones para agregar el batch y los canales (1 para escala de grises)
        img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión de lote
        img_array = np.expand_dims(img_array, axis=-1)  # Agregar dimensión de canales

        # Retornar el array procesado como respuesta (puedes adaptarlo según tu necesidad)
        return jsonify({"instances": img_array.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/predictEmotions', methods=['POST'])
def predict_emotions():
    try:
        # Obtener la imagen procesada desde la solicitud
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No se proporcionó ninguna imagen procesada"}), 400

        # Convertir los datos procesados a un array de NumPy
        img_array = np.array(data['image'], dtype=np.float32)

        # --------------------------API--------------------------
        try:
            # Preparar los datos para la API de predicción
            data = json.dumps({"signature_name": "serving_default", "instances": img_array.tolist()})
            headers = {'Content-Type': 'application/json'}
            json_response = requests.post(
                'https://tfexpressions-v1.onrender.com/v1/models/saved_model/versions/2:predict',
                data=data,
                headers=headers,
                verify=False
            )
            json_response.raise_for_status()  # Verificar si la solicitud fue exitosa

            # Obtener la predicción y mapearla a la emoción correspondiente
            predictions = json.loads(json_response.text)['predictions']
            facial_express = np.argmax(predictions, axis=1)
            emotion_label = label_to_text[int(facial_express[0])]  # Convertir a entero y obtener la emoción
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"No se pudo establecer una conexión con el servidor de TensorFlow Serving. Detalles: {e}"}), 500
        # --------------------------API--------------------------

        # Retornar la emoción como respuesta
        return jsonify({"emotion_label": emotion_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)