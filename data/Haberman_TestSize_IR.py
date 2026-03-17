import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from data.common.change_rate_data import change_rate_data
# from common.change_rate_data import change_rate_data

def load_data(test_size, new_rate=0.2):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/haberman.csv')
    dataset_desc = dataset.describe(include='all')
    haberman_map = {2: 1, 1: -1}
    dataset['class'] = dataset['class'].map(haberman_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 3].values

    X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train,y_train, X_test, y_test






