FROM python:3.8

WORKDIR /fast_api_server

COPY requirements.txt fast_api_server.py ./model_files .

RUN sudo apt-get install python3-opencv

RUN pip install -r requirements.txt

CMD ["uvicorn", "fast_api_server:app --host 0.0.0.0 --port 5000"]
