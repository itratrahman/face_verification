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

cosine_loss = CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)

class request_body(BaseModel):
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

def extract_face(img, dim = (160,160)):
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

@app.post('/predict')
def predict(data : request_body):

    response = {"success": False}
    image_1_base64 = request_body.image_1
    image_2_base64 = request_body.image_2

    try:
        image_1 = Image.open(BytesIO(base64.b64decode(passport)))
        response["face_1"] = "could read image 1"
    except:
        response["face_1"] = "could not read image 1"
        image_1 = None

    try:
        image_2 = Image.open(BytesIO(base64.b64decode(passport)))
        response["face_2"] = "could read image 2"
    except:
        response["face_2"] = "could not read image 2"
        image_2 = None

    try:
        if image_1 is not None:
            face_1 = extract_face(image_1, dim=(160, 160))
            response["face_1"] = "could detect a single face in image 1"
    except:
        response["face_1"] = "could not detect a single face in image 1"

    try:
        if image_2 is not None:
            face_2 = extract_face(image_2, dim=(160, 160))
            response["face_1"] = "could detect a single face in image 2"
    except:
        response["face_1"] = "could not detect a single face in image 2"

    if (type(face_1) is np.ndarray) and (type(face_2) is np.ndarray):
        face_1 = face_1[np.newaxis,:,:,:]
        face_2 = face_1[np.newaxis,:,:,:]
        face_1_embed = embedder(face_1)[np.newaxis,:]
        face_2_embed = embedder(face_2)[np.newaxis,:]
        loss = cosine_loss(face_1_embed, face_2_embed)[0]
        if loss<=-0.363564:
            pred = 1
        else:
            pred = 0
        response["prediction"] = pred
        # indicate that the request was a success
        response["success"] = True

    return response
