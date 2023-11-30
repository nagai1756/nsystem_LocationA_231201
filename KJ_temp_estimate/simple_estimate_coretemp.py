import numpy as np
import pandas as pd
import glob as gl
import re
from sklearn.svm import SVR
import os
import time
import pickle


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


class extimate_coretemp:
    def __init__(self, load_model=False):
        root_path = os.path.dirname(os.path.abspath(__file__))
        if load_model:
            with open(f"{root_path}/estimate_coretemp_model2.pkl", "rb") as file:
                self.model = pickle.load(file)
        else:
            # =================== PATH =====================
            affine_path_train = f"{root_path}/dataset2/*.npy"
            csv_path = f"{root_path}/coretemp_dataset.csv"
            numdata_train = 54
            # ==================== DATA ==================
            # Axillary temperature data acquisition
            datalist = pd.read_csv(csv_path)
            temp_data_train = datalist["coretemp2"].values
            # temp_data_test = datalist['coretemp_t'].values
            # Remove rows containing NaN values in temp_data
            nan_indices = np.isnan(temp_data_train)
            temp_data_train = temp_data_train[~nan_indices]
            # Acquisition of facial thermal image data for training
            thermal_path_train = gl.glob(affine_path_train)
            thermal_path_train = sorted(thermal_path_train, key=natural_keys)
            # Initialize the array to store thermal images for training
            thermal_imgs_flat_train = np.zeros((numdata_train, 120 * 120))
            # Load and reshape thermal image data for training
            for i, file in enumerate(thermal_path_train):
                thermal_imgs = np.load(file)
                thermal_imgs_flat_train[i] = thermal_imgs.reshape(-1)
            # ================== MAIN =====================
            # model building
            # Initialize support vector machine regression model
            self.model = SVR(kernel="poly", C=1, degree=4)
            X_train, y_train = thermal_imgs_flat_train, temp_data_train
            self.model.fit(X_train, y_train)
            with open(f"{root_path}/estimate_coretemp_model2.pkl", "wb") as file:
                pickle.dump(self.model, file)

    def estimate(self, thermo_affine_temp):
        X_test = thermo_affine_temp.reshape(1, -1)
        y_pred = round(self.model.predict(X_test)[0], 1)
        # if y_pred >= 37.0:
        #     y_pred = 37.0

        # if y_pred <= 35.2:
        #     y_pred = 35.2
        return y_pred
