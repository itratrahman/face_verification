# FACE VERIFICATION
Face verification is the task of comparing a candidate face to another, and verifying whether it is a match. It is a one-to-one mapping: you have to check if this person is the correct one. This project uses deep learning model for face verification. This project test performance of trained and pretrained model on benchmark data.

## 1. Folder Description
- `data`- contains benchmark datasets each contained in designated folders.
- `models`- contains pretrained model files

## 2. Installation Procedure
This project is run on ubuntu. Replicate environment using: `conda env create -f environment.yml`

## 3. File Description
- `generate_face_pairs.py` - this python script generates positive and negative pairs for face verification using vgg_face_2 test set.
- `cleanup.sh` - this bash script cleans up the numpy arrays which stores the face pairs.
- `test_facenet_vgg_test_set.ipynb` - this noteboook tests the performance of facenet on pairs generated from vgg_face_2 test set.
