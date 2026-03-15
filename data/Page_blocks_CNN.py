import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.utils import to_categorical
#from common.change_rate_data import change_rate_data

def load_data(test_size):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/page-blocks.csv')
    dataset_desc = dataset.describe(include = 'all')
    pageblocks_map = {5 : 1, 1: 0, 2: 0, 3: 0, 4: 0}
    dataset['class'] = dataset['class'].map( pageblocks_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 10].values


    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    train_labels = to_categorical(y_train,num_classes=2)
    test_labels = to_categorical(y_test,num_classes=2)

    
    return X_train, train_labels, X_test, test_labels