import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data.common.change_rate_data import change_rate_data

def load_data(test_size, new_rate):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/seismic_bumps.csv')
    Dataset_map = {1 : 1, 0: -1}
    dataset['class'] = dataset['class'].map(Dataset_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 18].values
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    onehotencoder_X.fit_transform(X).toarray()   
    X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    onehotencoder_X.fit_transform(X).toarray()
    X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    onehotencoder_X.fit_transform(X).toarray()
    X[:, 7] = labelencoder_X.fit_transform(X[:,7])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    onehotencoder_X.fit_transform(X).toarray()
    X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42,stratify=y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 1)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    pca = PCA(n_components =11)
    X_train  = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)


    return X_train, y_train, X_test, y_test

# X_train, y_train, X_test, y_test = load_data()
