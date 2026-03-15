import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from data.common.change_rate_data import change_rate_data, change_rate_data_cnn



def load_data(test_size,new_rate):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/transfusion.csv')
    transfusion_map = {1:1, 0:-1}
    dataset['whether he/she donated blood in March 2007'] = dataset['whether he/she donated blood in March 2007'].map(transfusion_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    #Split data
    X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = tts(X, y, test_size = test_size, random_state = 42, stratify=y)
    #Scalling Data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    return X_train, y_train, X_test, y_test

