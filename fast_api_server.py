import os
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity
from keras_facenet import FaceNet
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import uvicorn
from io import BytesIO
import base64

# base directory
BASE_DIR = os.path.abspath(os.path.dirname("__file__"))
# filepath of haarcascade face detection classifier model
HAARCASCADE_MODEL_PATH = os.path.join(BASE_DIR,
                                      "model_files",
                                      "haarcascade_frontalface_default.xml")
# tf cosine similarity object
cosine_loss = CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)

class request_body(BaseModel):
    """
    pydantic post request parameters
    """
    image_1: str
    image_2: str

# fastapi app
app = FastAPI()

def l2_normalize(x, axis=-1, epsilon=1e-10):
    """
    A function to l2 normalize an array
    """
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

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

# load facenet and face detector model
load_facenet()
load_face_detector()

@app.post('/predict')
def predict(data : request_body):
    """
    Post request which serves as prediction engine
    """
    # response dictionary
    response = {"success": False}
    # post request parameters
    image_1_base64 = data.image_1
    image_2_base64 = data.image_2

    try:
        # read image of face 1 which is encoded in base64
        image_1 = Image.open(BytesIO(base64.b64decode(image_1_base64)))
        response["face_1"] = "could read image 1"
    except:
        response["face_1"] = "could not read image 1"
        image_1 = None

    try:
        # read image of face 2 which is encoded in base64
        image_2 = Image.open(BytesIO(base64.b64decode(image_2_base64)))
        response["face_2"] = "could read image 2"
    except:
        response["face_2"] = "could not read image 2"
        image_2 = None

    try:
        # extract face from hte image
        if image_1 is not None:
            face_1 = extract_face(image_1, dim=(160, 160))
            response["face_1"] = "could detect a single face in image 1"
        else:
            face_1 = None
    except:
        face_1 = None
        response["face_1"] = "could not detect a single face in image 1"

    try:
        # extract face from the image
        if image_2 is not None:
            face_2 = extract_face(image_2, dim=(160, 160))
            response["face_2"] = "could detect a single face in image 2"
        else:
            face_2 = None
    except:
        face_2 = None
        response["face_2"] = "could not detect a single face in image 2"

    # if both face1 and face2 are numpy arrays
    if (type(face_1) is np.ndarray) and (type(face_2) is np.ndarray):

        # make the images 4D
        face_1 = face_1[np.newaxis,:,:,:]
        face_2 = face_2[np.newaxis,:,:,:]
        # l2 normalize the images of faces
        # and then compute embeddings using facenet model
        face_1_embed = l2_normalize(embedder.embeddings(face_1))[0]
        face_2_embed = l2_normalize(embedder.embeddings(face_2))[0]
        # compute the cosine loss of the embeddings of faces
        loss = cosine_loss([face_1_embed], [face_2_embed])
        # calculate prediction based on cosine loss
        if loss<=-0.363564:
            pred = 1
        else:
            pred = 0
        # set the prediction in request response
        response["prediction"] = pred
        # indicate that the request was a success
        response["success"] = True

    return response

if __name__ == "__main__":
    uvicorn.run(app, port=5000, host="0.0.0.0")
