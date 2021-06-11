# FACE VERIFICATION
Face verification is the task of comparing a candidate face to another, and verifying whether it is a match. It is a one-to-one mapping: you have to check if this person is the correct one. This project uses deep learning model for face verification. This project test performance of trained and pretrained model on benchmark data.

![Face Verification Image](https://github.com/itratrahman/face_verification/blob/experimental/image/IMAGE.png?raw=true)

## 1. Folder Description
- `data`- contains benchmark datasets each contained in designated folders.
- `models`- contains pretrained model files
- `image` - contains static image for the readme

## 2. Installation Procedure
This project is run on ubuntu using python 3. Installation is preferably done in one of the 2 following ways below. Installation is for cpu only. Virtualenv installation is just for to run the post server only; running jupyter notebooks requires the conda installation.
### 2.1 Conda
In conda you can replicate environment using: `conda env create -f environment.yml`. The conda environment is in python 3.8
### 2.1 Virtualenv
Use the following commands to create the python environment using virtualenv:
- `sudo apt install python3-pip`
- `sudo pip3 install virtualenv`
- `virtualenv venv`
- `source venv/bin/activate`
- `sudo apt-get install python3-opencv`
- `pip install -r requirements.txt`

`NOTE`-Virtualenv installation is just for to run the fastapi post server only.

## 3. File Description
- `generate_face_pairs.py` - this python script generates positive and negative pairs for face verification using vgg_face_2 test set.
- `cleanup.sh` - this bash script cleans up the numpy arrays which stores the face pairs.
- `test_facenet_vgg_test_set.ipynb` - this noteboook tests the performance of facenet on pairs generated from vgg_face_2 test set.
- `fast_api_server.py` - fasFastAPI post server

## 4. Run the FastAPI server
The FastAPI server is a post server. Use the following command to rung the post server using the following command: `python fast_api_server.py`
