
FROM tensorflow/tensorflow:2.13.0-gpu

COPY . /home/app
WORKDIR /home/app/application
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD ["uvicorn", "--host", "0.0.0.0", "app:app", "--reload"]

