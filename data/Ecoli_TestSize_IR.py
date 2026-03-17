import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from data.common.change_rate_data import change_rate_data, change_rate_data_cnn

def load_data(test_size, new_rate=0.2):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/ecoli_new.csv')
    dataset_desc = dataset.describe(include = 'all')
    ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
    dataset['class'] = dataset['class'].map(ecoli_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 7].values
    X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train,y_train, X_test, y_test

# X, y = load_data()
# print(y)