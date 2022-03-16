import random

# import matplotlib
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2



from sklearn.metrics import classification_report,confusion_matrix
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential




#
# def face_detector(img_path):
#     img = cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray)
#     return len(faces) > 0
#
# from keras.preprocessing import image
# from tqdm import tqdm
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# def path_to_tensor(img_path):
#     img = image.load_img(img_path, target_size=(224, 224)
#     return np.expand_dims(x, axis=0)
#
# def paths_to_tensor(img_paths):
#     list_of_tensors = [path_to_tensor(img_path) for img_path in       tqdm(img_paths)]
#     return np.vstack(list_of_tensors)
#
# def ResNet50_predict_labels(img_path):
#     img = preprocess_input(path_to_tensor(img_path))
#     return np.argmax(ResNet50_model.predict(img))
#
# def dog_detector(img_path):
#     prediction = ResNet50_predict_labels(img_path)
#     return ((prediction <= 268) & (prediction >= 151))
#
#
#
#
# model = Sequential()model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224,224,3)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=32, kernel_size=2 , padding='same' , activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=64 , kernel_size=2 , padding='same' , activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.4))
# model.add(Conv2D(filters=128 , kernel_size=2 , padding='same' , activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.4))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(133,activation='softmax'))
# model.summary()