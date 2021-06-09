FROM ubuntu:18.04

WORKDIR /fast_api_server

COPY requirements.txt fast_api_server.py ./model_files ./

RUN apt update -y && \
    apt-get install python3 -y

RUN apt-get install python3-opencv -y

RUN apt-get install python3-pip -y

RUN pip3 install -r requirements.txt

CMD ["python", "fast_api_server.py"]
