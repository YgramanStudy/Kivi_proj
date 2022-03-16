import random

# import matplotlib
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2

from tensorflow import keras
# from tensorflow.keras import layers, Dense, Input, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from keras.layers import *
from keras.models import *
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report,confusion_matrix



IMG_WIDTH=224
IMG_HEIGHT=224
img_folder=r'Project data\Kaggle\1\training_set\training_set\dogs'

def show_random_data(img_folder):
    for i in range(5):
        file = random.choice(os.listdir(img_folder))
        image_path = os.path.join(img_folder, file)
        img = cv2.imread(image_path)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow(file, img)
    cv2.waitKey(0)
    cv2.destroyWindow()





img_folder=r'Project data\Kaggle\1\training_set\training_set\\'
test_img_folder=r'Project data\Kaggle\3\archive\dataset\test_set\\'

def create_dataset(img_folder):
    img_data_array = []
    class_name = []
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    print("END")
    return img_data_array, class_name  # extract the image array and class name



print("--------------------------------TRAIN------------------------------------")
img_data, class_name = create_dataset(img_folder)
target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
print("--------------------------------TEST------------------------------------")
test_img_data, test_class_name = create_dataset(test_img_folder)
test_target_dict = {k: v for v, k in enumerate(np.unique(test_class_name))}
test_target_val = [test_target_dict[test_class_name[i]] for i in range(len(test_class_name))]
print("--------------------------------DONE------------------------------------")

#pre-trained on the ImageNet dataset
base_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3), include_top = False, weights = "imagenet")
#freeze our base model from being updated during training
base_model.trainable = False

model = tf.keras.Sequential([base_model,
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(2, activation="softmax")
                                ])

base_learning_rate = 0.00001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])


# print(np.array(img_data, np.float32).shape),
# print(np.array(test_img_data, np.float32).shape)
# print(np.array(list(map(int,target_val)), np.float32).shape)
# print(np.array(list(map(int,test_target_val)), np.float32).shape)







history = model.fit(np.array(img_data, np.float32),np.array(list(map(int,target_val)), np.float32),epochs = 500 , validation_data = (np.array(test_img_data, np.float32), np.array(list(map(int,test_target_val)), np.float32)))

predictions = model.predict_classes(np.array(test_img_data, np.float32))
predictions = predictions.reshape(1,-1)[0]

print(classification_report(np.array(list(map(int,test_target_val)), np.float32), predictions, target_names = ['cats (Class 0)','dogs (Class 1)']))