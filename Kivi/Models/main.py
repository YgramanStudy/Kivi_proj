from .model_export_inport import model_import
from .get_data import preper_data
from .model2 import model2
import numpy as np
import pickle

train_img_folder = r'Project data\Kaggle\   3\archive\dataset\training_set\\'
test_img_folder = r'Project data\Kaggle\3\archive\dataset\test_set\\'
filename = "finalized_model.sav"

def run_and_test(train, test, model_filename):
    x_train, y_train, x_val, y_val = preper_data(train, test)
    history = model2(x_train, y_train, x_val, y_val, model_filename)
    model = model_import(model_filename)
    predict_x = model.predict(x_val)
    classes_x = np.argmax(predict_x, axis=1)
    print(classes_x)
    return


run_and_test(train_img_folder, test_img_folder, filename)
