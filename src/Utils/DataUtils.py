import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
import src.Config as Config
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

__all__ = ["load_arff", "load_dataset", "process_data"]

DATASET_ROOT = "./datasets"

def load_dataset(dataset='???', seed=998244353):
    return load_arff(dataset, seed=seed)

def load_arff(dataset="???", seed=998244353):
    path = DATASET_ROOT
    with open(os.path.join(path, f"{dataset}_TRAIN.arff"), 'r', encoding='utf-8') as file:
        train_data = loadarff(file)[0]
    with open(os.path.join(path, f"{dataset}_TEST.arff"), 'r', encoding='utf-8') as file:
        test_data = loadarff(file)[0]
    
    # Extract data from arff format
    def extract_data(data):
        res_data, res_label = [], []
        for row in data:
            row = list(row)
            data = np.array(row[: -1]).reshape(1, -1, 1)
            label = np.array(row[-1].decode("utf-8")).reshape(-1, 1)
            res_data.append(data)
            res_label.append(label)
        res_data = np.vstack(res_data)
        res_label = np.vstack(res_label).ravel()
        return res_data, res_label
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    # Resplit & modify label training & testing data
    # Follow the setting in SREA: https://github.com/Castel44/SREA/blob/main/src/utils/ucr_datasets.py
    X = np.concatenate((train_X, test_X))
    y = np.concatenate((train_y, test_y))
    y = LabelEncoder().fit_transform(y)
    assert (y.min() == 0)
    
    # Output dataset information
    print("Dataset -", dataset)
    print(" * Label Set", np.unique(train_y))
    
    return X, y

def process_data(X, y, clean_y, seed=998244353, test_size=0.2, valid_size=0.1):
    tv_idx, test_idx = train_test_split(range(X.shape[0]), stratify=clean_y, test_size=test_size, random_state=seed)
    
    test_X, test_y = X[test_idx, :, :], clean_y[test_idx]
    tv_X, tv_y, clean_tv_y = X[tv_idx, :, :], y[tv_idx], clean_y[tv_idx]
    
    scaler = TimeSeriesScalerMeanVariance()
    tv_X = scaler.fit_transform(tv_X)
    test_X  = scaler.transform(test_X)
    if np.isnan(tv_X).any(): tv_X = np.nan_to_num(tv_X, copy=False, nan=0.0)
    if np.isnan(test_X).any():  test_X  = np.nan_to_num(test_X , copy=False, nan=0.0)

    train_idx, valid_idx = train_test_split(range(tv_X.shape[0]), test_size=valid_size, random_state=seed, stratify=clean_tv_y)
    
    train_X, valid_X = tv_X[train_idx, :, :], tv_X[valid_idx, :, :]
    train_y, valid_y = tv_y[train_idx], tv_y[valid_idx]
    clean_train_y, clean_valid_y = clean_tv_y[train_idx], clean_tv_y[valid_idx]
    
    # Output dataset information    
    print(" * Training Set", train_X.shape, train_y.shape)
    print(" * Training Set", valid_X.shape, valid_y.shape)
    print(" * Testing  Set", test_X.shape, test_y.shape)

    return train_X, train_y, clean_train_y, valid_X, valid_y, clean_valid_y, test_X, test_y