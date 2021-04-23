import random
import os
import numpy as np
import pandas as pd
import cv2
import progressbar

BASE_DIR = os.path.abspath(os.path.dirname("__file__"))
VGG_TEST_SET_DIR = os.path.join(BASE_DIR, "data","vgg_face_2", "test_set")
NUMPY_DIR = os.path.join(BASE_DIR, "data","numpy_arrays")
HAARCASCADE_MODEL_PATH = os.path.join(BASE_DIR, "model_files", "haarcascade_frontalface_default.xml")
RANDOM_SEED = 1
NUMPY_SEED = 1

def face_detect(image,
                dim = (160,160)):
    '''
    Detect to detect and return face
    '''
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))
    if len(faces) != 1:
        return None
    (x, y, w, h) = faces[0]
    area_proportion = (w*h)/(gray_img.shape[0]*gray_img.shape[1])
    if area_proportion<=0.20:
        return None
    cropped_img = image[y:y+h, x:x+w]
    if dim:
        cropped_img = cv2.resize(cropped_img, dim, interpolation = cv2.INTER_AREA)

    return cropped_img

def create_batch(batch_size, face_dir, face_list, dim = (160, 160)):

    """
    Functin to create batch of image pairs
    """
    postive_size = batch_size//2
    negative_size = batch_size - (batch_size//2)
    max_value = postive_size + negative_size
    X1_positives = np.zeros(shape = (postive_size,dim[0],dim[1], 3), dtype = "float32")
    X2_positives = np.zeros(shape = (postive_size,dim[0],dim[1], 3), dtype = "float32")
    X1_negatives = np.zeros(shape = (negative_size,dim[0],dim[1], 3), dtype = "float32")
    X2_negatives = np.zeros(shape = (negative_size,dim[0],dim[1], 3), dtype = "float32")

    with progressbar.ProgressBar(max_value=max_value) as bar:
        bar_counter = 0
        counter = 0
        while counter<postive_size:
            random_person = np.random.choice(face_list)
            random_person_dir = os.path.join(face_dir,random_person)
            photos = os.listdir(random_person_dir)
            photo1 = np.random.choice(photos)
            photo2 = np.random.choice(photos)
            while photo1 == photo2:
                photo1 = np.random.choice(photos)
                photo2 = np.random.choice(photos)
            dir1= os.path.join(random_person_dir,photo1)
            dir2= os.path.join(random_person_dir,photo2)
            try:
                photo1 = face_detect(cv2.imread(dir1))
                photo2 = face_detect(cv2.imread(dir2))
                if np.isnan(photo1).any():
                    5/0
                if np.isnan(photo2).any():
                    5/0
                X1_positives[counter] = photo1
                X2_positives[counter] = photo2
                counter +=1
                bar.update(bar_counter)
                bar_counter += 1
            except:
                continue

        counter = 0
        while counter<negative_size:
            random_person_1 = np.random.choice(face_list)
            random_person_2 = np.random.choice(face_list)
            while random_person_1 == random_person_2:
                random_person_1 = np.random.choice(face_list)
                random_person_2 = np.random.choice(face_list)
            random_person_1_dir = os.path.join(face_dir,random_person_1)
            random_person_2_dir = os.path.join(face_dir,random_person_2)
            photos1 = os.listdir(random_person_1_dir)
            photos2 = os.listdir(random_person_2_dir)
            photo1 = np.random.choice(photos1)
            photo2 = np.random.choice(photos2)
            dir1= os.path.join(random_person_1_dir,photo1)
            dir2= os.path.join(random_person_2_dir,photo2)
            try:
                photo1 = face_detect(cv2.imread(dir1))
                photo2 = face_detect(cv2.imread(dir2))
                if np.isnan(photo1).any():
                    5/0
                if np.isnan(photo2).any():
                    5/0
                X1_negatives[counter] = photo1
                X2_negatives[counter] = photo2
                counter +=1
                bar.update(bar_counter)
                bar_counter += 1
            except:
                continue

    X1 = np.concatenate([X1_positives, X1_negatives], axis = 0).astype('float32')
    X2 = np.concatenate([X2_positives, X2_negatives], axis = 0).astype('float32')
    y = np.array([1 for i in range(postive_size)] +
                 [0 for i in range(negative_size)]).astype('int32').reshape(-1,1)
    return X1, X2, y

if __name__ == "__main__":

    print("######################### GENERATE AND SAVE IMAGE PAIRS #########################\n")

    random.seed(RANDOM_SEED)
    np.random.seed(NUMPY_SEED)

    face_detector = cv2.CascadeClassifier(HAARCASCADE_MODEL_PATH)

    face_list = os.listdir(VGG_TEST_SET_DIR)

    X1, X2, y = create_batch(1024*4, VGG_TEST_SET_DIR, face_list)

    if not os.path.exists(NUMPY_DIR):
        os.mkdir(NUMPY_DIR)

    np.save(os.path.join(NUMPY_DIR, "X1.npy"), X1)
    np.save(os.path.join(NUMPY_DIR, "X2.npy"), X2)
    np.save(os.path.join(NUMPY_DIR, "y.npy"), y)

    print("\n#################################################################################\n")
