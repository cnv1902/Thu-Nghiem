import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data.common.change_rate_data import change_rate_data

def load_data(test_size, new_rate):
    #read data from csv
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/Spect_Heart.csv')
    Dataset_map = {1 : -1, 0: 1}
    dataset['OVERALL_DIAGNOSIS'] = dataset['OVERALL_DIAGNOSIS'].map(Dataset_map) 
    #tranfer to feature and label
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, 22].values
    # X = dataset.values[:, 0:-1]
    # y = dataset.values[:, 22]

    # Splitting the dataset into trainig and test set
    
    X, y = change_rate_data(X, y , new_rate = new_rate)    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42,stratify=y)
    
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 1)
    return X_train, y_train, X_test, y_test
