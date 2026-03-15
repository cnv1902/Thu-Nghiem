import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from ..common.change_rate_data import  change_rate_data
from data.common.change_rate_data import change_rate_data
# def load_data(test_size, new_rate):
#     data = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/LEC7/madaboost_25112021/madaboost/data/datasets/Vertebral_column.csv')
#     diag_map = {'Abnormal': -1.0, 'Normal': 1.0}
#     data['Label class'] = data['Label class'].map(diag_map)
#     X = data.values[:, 0:-1]
#     y = data.values[:, 6]
#     X, y = change_rate_data(X, y , new_rate = new_rate)
    
#     X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
#     # X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42)
#     return X_train,y_train, X_test, y_test
def load_data():
    data = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/Vertebral_column.csv')
    diag_map = {'Abnormal': -1.0, 'Normal': 1.0}
    data['Label class'] = data['Label class'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 6]
    X = np.array(X)
    y = np.array(y)
    return X, y
