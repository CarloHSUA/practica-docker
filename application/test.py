import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

def prepare_data(x):
  x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)    # Redimensionamos para añadir el canal
  x = x.astype(np.float32)      # TODO: Transforma la variable "x" a decimal
  x /= 255      # TODO: Normaliza la variable "x" entre 0 y 1
  return x

def leer_imagen_desde_url(url):
    try:
        # Hacer una solicitud GET para obtener la imagen desde la URL
        respuesta = requests.get(url)
        
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

            # print(img.shape)

            return img
        else:
            print("No se pudo obtener la imagen. Código de estado:", respuesta.status_code)
            return None
    except Exception as e:
        print("Ocurrió un error al obtener la imagen:", str(e))
        return None