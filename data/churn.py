import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data.common.change_rate_data import change_rate_data

def load_data(test_size,new_rate):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/churn.csv')
    dataset_desc = dataset.describe(include = 'all')
    Churn_map = {'False.' : -1, 'True.': 1}
    dataset['Churn?'] = dataset['Churn?'].map(Churn_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 20].values
    labelencoder_X = LabelEncoder()
    # Encoding the State Categorization
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    onehotencoder_X = OneHotEncoder()
    onehotencoder_X.fit_transform(X).toarray()
    # Encoding the Phone Categorization
    X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
    onehotencoder_X = OneHotEncoder()
    onehotencoder_X.fit_transform(X).toarray()
    # Encoding the Int'l Plan Categorization
    X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
    onehotencoder_X = OneHotEncoder()
    onehotencoder_X.fit_transform(X).toarray()
    # Encoding the Int'l VMail Plan Categorization
    X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
    onehotencoder_X = OneHotEncoder()
    onehotencoder_X.fit_transform(X).toarray()
    # Encoding the ...n Categorization
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    onehotencoder_X = OneHotEncoder()
    onehotencoder_X.fit_transform(X).toarray()
    #Split data
    X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = tts(X, y, test_size = test_size, random_state = 42, stratify=y)
    #Scalling Data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    #Analys data
    pca = PCA(n_components = 15)
    X_train  = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, y_train, X_test, y_test