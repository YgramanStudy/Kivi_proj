import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator


from sklearn.metrics import classification_report,confusion_matrix
from .model_export_inport import model_export
import tensorflow as tf
from .model_export_inport import model_export
import cv2
import os

import numpy as np


def model2(x_train, y_train, x_val, y_val,model_filename):
    print("model2 start")
    img_size = 224
    base_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3), include_top = False, weights = "imagenet")
    base_model.trainable = False

    model = tf.keras.Sequential([base_model,
                                     tf.keras.layers.GlobalAveragePooling2D(),
                                     tf.keras.layers.Dropout(0.2),
                                     tf.keras.layers.Dense(1, activation='sigmoid')

                                    ])



    base_learning_rate = 0.00001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])





    model_export(model, model_filename)
    print("model2 DONE")
    return history




# predictions = model.predict_classes(x_val)
# predictions = predictions.reshape(1,-1)[0]



# print(classification_report(y_val, predictions, target_names = ['cats (Class 0)','dogs (Class 1)']))










