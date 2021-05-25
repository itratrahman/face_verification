import os
import numpy as np
import cv2
from tensorflow.keras.losses import CosineSimilarity
from keras_facenet import FaceNet
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from io import BytesIO
import base64

BASE_DIR = os.path.abspath(os.path.dirname("__file__"))
HAARCASCADE_MODEL_PATH = os.path.join(BASE_DIR, "model_files", "haarcascade_frontalface_default.xml")

class Item(BaseModel):
    image_1: str
    image_2: str

app = FastAPI()

def load_facenet():
    """
    A function to load keras facenet
    """
    global embedder
    embedder = FaceNet()

def load_face_detector():
    """
    A function to load opencv face detector
    """
    global face_detector
    face_detector = cv2.CascadeClassifier(HAARCASCADE_MODEL_PATH)

def extract_face_(img, face):
    """
    A function to extract face
    """
    roi = img.copy()
    (x, y, w, h) = face
    face_img = roi[y: y + h, x: x + w]
    return face_img

def extract_face(img, dim = (256,256)):
    '''
    Detect face
    '''
    img = np.asarray(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))
    if len(faces) == 1:
        face_img = extract_face_(img, faces[0])
        face_img = cv2.resize(face_img, dim, interpolation = cv2.INTER_AREA)
    return face_img

load_facene()
load_face_detector()
