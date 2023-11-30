docker build -t fashion:1 .
docker run --gpus all -it -p 8000:8000 fashion:1 /bin/bash
