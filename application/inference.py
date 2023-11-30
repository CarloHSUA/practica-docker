import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf # TODO - importa la librería de Tensorflow como "tf"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from test import leer_imagen_desde_url, prepare_data
import os
from io import BytesIO
import cv2

tf.random.set_seed(2)  # Fijamos la semilla de TF
np.random.seed(2)  # Fijamos la semilla


# Descargamos la base de datos
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

labels = ['Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso/a', 'Botín']

def prepare_data(x):
  x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)    # Redimensionamos para añadir el canal
  x = x.astype(np.float32)      # TODO: Transforma la variable "x" a decimal
  x /= 255      # TODO: Normaliza la variable "x" entre 0 y 1
  return x

print(x_train.shape, x_train.dtype)

x_train = prepare_data(x_train)
x_test  = prepare_data(x_test)

print(f"y_train:{y_train}")

# Transformamos las etiquetas a categórico (one-hot)
NUM_LABELS = 10
y_train = tf.keras.utils.to_categorical(y_train, NUM_LABELS)   # TODO: Transforma la variable "y_train" a categórica
y_test = tf.keras.utils.to_categorical(y_test, NUM_LABELS)    # TODO: Transforma la variable "y_test" a categórica

print(f"y_train:{y_train}")

model_path = './models/fashion_model.h5'

if not os.path.exists(model_path):

    # Para los primeros ejemplos vamos a limitar el número de imágenes de
    # entrenamiento a 50. Además nos guardamos un backup con todas las imágenes.
    x_train_backup = x_train.copy()
    y_train_backup = y_train.copy()
    x_train = x_train[:50]
    y_train = y_train[:50]


    # Mostramos (de nuevo) las dimensiones de los datos
    print('Datos para entrenamiento:')
    print(' - x_train: {}'.format( x_train.shape )) # TODO: Muestra la forma de la variable x_train
    print(' - y_train: {}'.format( y_train.shape )) # TODO: Muestra la forma de la variable y_train
    print('Datos para evaluación:')
    print(' - x_test: {}'.format( x_test.shape ))   # TODO: Muestra la forma de la variable x_test
    print(' - y_test: {}'.format( y_test.shape ))    # TODO: Muestra la forma de la variable y_test


    model2 = Sequential()

    print(x_train.shape[1:])
    # Capa convolucional con 64 filtros de tamaño 3x3 seguida de un MaxPooling de 2x2
    model2.add(Conv2D( 32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))   # TODO: Establece el número de filtros a 32
    model2.add( MaxPooling2D(pool_size=(2,2)) )   # TODO: Añade la capa de MaxPooling2D con la configuración indicada
    model2.add( Dropout(0.2) )   # TODO: Añade un Dropout de 0.2

    # Capa convolucional con 32 filtros de tamaño 3x3 seguida de un MaxPooling de 2x2
    model2.add(Conv2D( 32, (3, 3), activation='relu'))   # TODO: Establece el número de filtros a 32
    model2.add( MaxPooling2D(pool_size=(2,2)) )   # TODO: Añade la capa de MaxPooling2D con la configuración indicada
    model2.add( Dropout(0.2) )   # TODO: Añade un Dropout de 0.2

    # Capa Fully Connected
    model2.add( Flatten() )  # TODO: Añade una capa tipo Flatten
    model2.add(Dense(NUM_LABELS, activation='softmax'))

    print(model2.summary())

    # Compilamos y entrenamos
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )


    # Restauramos todas las imágenes de entrenamiento
    x_train = x_train_backup
    y_train = y_train_backup

    print('Datos para entrenamiento:')
    print(' - x_train: {}'.format( x_train.shape ))
    print(' - y_train: {}'.format( y_train.shape ))


    # Iniciamos el entrenamiento
    history = model2.fit(x_train, y_train, validation_data=(x_test, y_test),
                        batch_size= 32,  # TODO: Establece el tamaño de batch a 32
                        epochs= 10,      # TODO: Establece el número de épocas a 10
                        verbose=1)



    # Evaluamos usando el test set
    score = model2.evaluate(x_test, y_test, verbose=0)

    print('Resultado en el test set:')
    print('Test loss: {:0.4f}'.format(score[0]))
    print('Test accuracy: {:0.2f}%'.format(score[1] * 100))


    # Evaluamos usando el test set
    score = model2.evaluate(x_test, y_test, verbose=0)

    print('Resultado en el test set:')
    print('Test loss: {:0.4f}'.format(score[0]))
    print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

    model2.save(model_path)

else:
   model2 =  tf.keras.models.load_model(model_path)

def testing_model():

  

  url = 'https://www.bluebananabrand.com/cdn/shop/files/ClassicTeeAqua1_1_720x.jpg?v=1683889376'
  url = 'https://img.kwcdn.com/product/1e23314fea/95c761bb-1ade-4300-82ef-b3dd6bc59563_800x800.jpeg'

  # url = input("URL: ")

  img = leer_imagen_desde_url(url)

  print(img.shape)
  # Ejecutamos la red
  prediction = model2.predict(img)   # TODO: Llama a la función para calcular la predicción a partir de la variable de entrada "img"

  print('Predicción: ', labels[np.argmax(prediction)])

# if __name__ == '__main__':
#    testing_model()
