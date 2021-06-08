FROM python:3.8

WORKDIR /fast_api_server

COPY requirements.txt fast_api_server.py ./model_files .

RUN yes | sudo apt-get install python3-opencv

RUN pip install -r requirements.txt

CMD ["python", "fast_api_server.py"]
