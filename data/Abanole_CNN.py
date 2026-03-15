import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.utils import to_categorical
#from common.change_rate_data import change_rate_data

def load_data(test_size):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/abalone.csv')
    dataset_desc = dataset.describe(include = 'all')
    abalone_map = {15 : 1, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0,
    11 : 0, 12 : 0, 13 : 0, 14 : 0, 16 : 0, 17 : 0, 18 : 0, 19 : 0, 20 : 0, 21 : 0, 22 : 0, 23 : 0,
    24 : 0, 25 : 0, 26 : 0, 27 : 0, 28 : 0, 29 : 0}
    dataset['Rings'] = dataset['Rings'].map(abalone_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    labelencoder_X = LabelEncoder()
    # Encoding the Sex Categorization
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore',dtype = np.float64)
    onehotencoder_X.fit_transform(X).toarray()

    # X = np.array(X)
    # y = np.array(y)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    X_train = np.asarray(X_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    train_labels = to_categorical(y_train,num_classes=2)
    test_labels = to_categorical(y_test,num_classes=2)

    # print(train_labels)
    # print(test_labels)
    print(X_train)
    print(X_test)
    
    return X_train, train_labels, X_test, test_labels


# load_data(0.2)