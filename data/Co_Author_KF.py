import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data.common.change_rate_data import change_rate_data


def load_data(): #new_rate
    data = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/Co_Author_100_500_1.csv')
    diag_map = {-1: 1.0, 1: -1.0}
    data['Label'] = data['Label'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 7]
    # X, y = change_rate_data(X, y , new_rate = new_rate)
    X = np.array(X)
    y = np.array(y)
    return X, y
    
