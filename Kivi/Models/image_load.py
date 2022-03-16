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


# plt.figure(figsize=(20,20))

# test_folder=r'CV\Intel_Images\seg_train\seg_train\forest'

IMG_WIDTH=200
IMG_HEIGHT=200
img_folder=r'Project data\Kaggle\1\training_set\training_set\dogs'
# print(os.listdir(img_folder))
def show_random_data(img_folder):
    for i in range(5):
        file = random.choice(os.listdir(img_folder))
        image_path = os.path.join(img_folder, file)
        img = cv2.imread(image_path)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow(file, img)
    cv2.waitKey(0)
    cv2.destroyWindow()

    # img = mpimg.imread(image_path)
    # ax = plt.subplot(1, 5, i+1)
    # ax.title.set_text(file)
    # # plt.imshow(img)
    # plt.imshow(img)
    # plt.show()



img_folder=r'Project data\Kaggle\1\training_set\training_set\\'
test_img_folder=r'Project data\Kaggle\3\archive\dataset\test_set\\'

def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            # print('1')
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


img_data, class_name = create_dataset(img_folder)
test_img_data, test_class_name = create_dataset(test_img_folder)
print("--------------------------------------------------------------------")

target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

test_target_dict = {k: v for v, k in enumerate(np.unique(test_class_name))}
test_target_val = [test_target_dict[test_class_name[i]] for i in range(len(test_class_name))]



model=tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6)
        ])
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=np.array(img_data, np.float32), y=np.array(list(map(int,target_val)), np.float32), epochs=5)


predictions = model.predict_classes(img_data)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(test_img_data, predictions, target_names = ['cats (Class 0)','dogs (Class 1)']))
# score = model.evaluate(np.array(test_img_data, np.float32), np.array(list(map(int,test_target_val)), verbose=1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])




# acc = history.history['accuracy']
# # val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# # val_loss = history.history['val_loss']
#
# epochs_range = range(500)
#
# plt.figure(figsize=(15, 15))
# plt.subplot(2, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# # plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()