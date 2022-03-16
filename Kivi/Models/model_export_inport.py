import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
from .model_export_inport import model_export


def model_export(model, filename):
    pickle.dump(model, open(filename, 'wb'))
    print("export done")


def model_import(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    print("import done")
    return loaded_model
