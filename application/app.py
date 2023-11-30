from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
import tensorflow as tf
import uvicorn
from test import leer_imagen_desde_url, prepare_data
import requests
from pydantic import BaseModel
import cv2
import os

class Url(BaseModel):
    url: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get_index():
    return FileResponse('static/index.html')

model_path = './models/fashion_model.h5'
# Cargar el modelo de Keras
if os.path.exists(model_path):
    modelo = load_model(model_path)  # Reemplaza 'ruta_al_modelo.h5' con la ruta real de tu modelo
    
labels = ['Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso/a', 'Botín']

@app.post("/predict/")
async def predict(url: Url):
    print(url.url)
    try:
        # Hacer una solicitud GET para obtener la imagen desde la URL
        respuesta = requests.get(url.url)
        
        # Verificar si la solicitud fue exitosa (código de estado 200)
        if respuesta.status_code == 200:
            # Leer la imagen desde la respuesta de la solicitud usando PIL
            imagen_pil = Image.open(BytesIO(respuesta.content))
            # Convertir la imagen PIL a una matriz numpy para usar con OpenCV
            # imagen_cv2 = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)

            # img = cv2.imread(url, cv2.IMREAD_COLOR)
            img = 255 - cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2GRAY)

            # Escalamos la imagen
            img = cv2.resize(img, (28, 28))

            # Normalizamos los datos
            # print(img.shape)
            img = prepare_data(np.array([img]))

            # Realizar la predicción
            prediction = modelo.predict(img)
            predicted_class = labels[np.argmax(prediction)]

            return {"predicted_class": predicted_class}
        else:
            return {"error": "No se pudo obtener la imagen. Código de estado: " + str(respuesta.status_code)}
    except Exception as e:
        return {"error": f"Error en la predicción: {str(e)}"}

if __name__ == "__main__":
    # Verificar si hay GPU disponible
    if tf.test.is_gpu_available():
        print("GPU está disponible")
        # Obtener información sobre la GPU
        gpu_devices = tf.config.list_physical_devices('GPU')
        print("Dispositivos GPU disponibles:", gpu_devices)
    else:
        print("No se detectó GPU, utilizando CPU")
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
