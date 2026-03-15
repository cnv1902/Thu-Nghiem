import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data.common.change_rate_data import change_rate_data


def load_data(test_size):
    data = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/Co-Author-100-500-st-unw.csv')
    # diag_map = {-1: -1.0, 1: -1.0}
    # data['Label'] = data['Label'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 7]
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    return X_train,y_train, X_test, y_test
