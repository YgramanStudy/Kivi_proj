import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator


from sklearn.metrics import classification_report,confusion_matrix
# from .model_export_inport import model_export
import tensorflow as tf
# from .model_export_inport import model_export
import cv2
import os
import pickle
import numpy as np



labels = ['cats', 'dogs']
img_size = 224
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

def preper_data(train_data_dir,test_data_dir):

    train = get_data(train_data_dir)
    val = get_data(test_data_dir)

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)

    print("Data is ready")
    return x_train, y_train, x_val, y_val

def model_export(model, filename):
    pickle.dump(model, open(filename, 'wb'))
    print("export done")


def model_import(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    print("import done")
    return loaded_model



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



    history = model.fit(x_train,y_train,epochs = 20 , validation_data = (x_val, y_val))

    model_export(model, model_filename)
    print("model2 DONE")
    return history


train_img_folder = r'Project data\Kaggle\3\archive\dataset\training_set\\'
test_img_folder = r'Project data\Kaggle\3\archive\dataset\test_set\\'
filename = "finalized_model.sav"

x_train, y_train, x_val, y_val = preper_data(train_img_folder, test_img_folder)
history = model2(x_train, y_train, x_val, y_val, filename)
model = model_import(filename)
predict_x = model.predict(x_val)
classes_x = np.argmax(predict_x, axis=1)
print(classes_x)





















# predictions = model.predict_classes(x_val)
# predictions = predictions.reshape(1,-1)[0]



# print(classification_report(y_val, predictions, target_names = ['cats (Class 0)','dogs (Class 1)']))
