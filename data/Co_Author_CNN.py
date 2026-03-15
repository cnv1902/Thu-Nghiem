import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data.common.change_rate_data import change_rate_data
from keras.utils import to_categorical

def load_data(test_size): #new_rate
    data = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/Co_Author_100_500_1.csv')
    diag_map = {-1: 0, 1: 1}
    data['Label'] = data['Label'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 7]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    train_labels = to_categorical(y_train,num_classes=2)
    test_labels = to_categorical(y_test,num_classes=2)

    return X_train, train_labels, X_test, test_labels

    
