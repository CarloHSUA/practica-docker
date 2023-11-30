# Práctica Docker
Práctica usando Docker para realizar inferencia en modelos de Deep Learning

## Requisitos
1. Tener instalada la versión de 12.3 de cuda, si no la tienes, descargala aquí: 
[Versión 12.3 de Cuda](https://developer.nvidia.com/cuda-downloads)

2. Tener instalado el Container Toolkit de Nvidia, si no lo tienes, descargala aquí: 
[Container Toolkit de Nvidia](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

2. Tener instalado Docker, si no lo tienes, descargalo aquí: 
[Docker para ubuntu](https://docs.docker.com/engine/install/ubuntu/)

En mi caso se ha instalado Docker para Ubuntu, pero puedes instalarlo para tu sistema operativo en particular. 

## Ejecución
**AVISO**: Antes de nada deberás tener espacio suficiente para crear el contenedor, en concreto 8GB más o menos.
Una vez realizados los requisitos, se procede a desplegar el contenedor de docker mediante el siguiente comando:
```
docker build -t fashion:1 .
docker run --gpus all -p 8000:8000 fashion:1
```

Esto lo que hará es crear una contenedor a partir del Dockerfile.

Una vez realizado lo anterior, ya puedes acceder al navegador con http://localhost:8000/

Si deseas entrenar el modelo en el contenedor de Docker, deberas ejecutar los siguientes comandos:
```
docker build -t fashion:1 .
docker run --gpus all -it -p 8000:8000 fashion:1 /bin/bash
```

Una vez dentro del contenedor puedes ejecutar lo siguiente:
```
python inference.py
```

Tras esto, ya puedes levantar el servidor con FastAPI de la siguiente forma:
```
uvicorn --host 0.0.0.0 app:app --reload
```
Una vez realizado lo anterior, ya puedes acceder al navegador con http://localhost:8000/

## Estructura del repositorio
- models/ : Contiene el modelo preentrenado con Keras en formato .h5.
- static/ : Contiene el archivo .html. 
