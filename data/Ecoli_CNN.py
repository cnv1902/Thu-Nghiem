import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.utils import to_categorical
#from common.change_rate_data import change_rate_data

def load_data(test_size):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/ecoli_new.csv')
    dataset_desc = dataset.describe(include = 'all')
    # ecoli_map = {' im':1.0, ' cp':-1.0, 'imL':-1.0,'imS':-1.0,'imU':-1.0,' om':-1.0,'omL':-1.0,' pp':-1.0}
    ecoli_map = {' im':1.0, ' cp':0, 'imL':0,'imS':0,'imU':0,' om':0,'omL':0,' pp':0}
    dataset['class'] = dataset['class'].map(ecoli_map)
    X = dataset.drop(['class'], axis=1)
    y = dataset['class']
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42,stratify=y)
    train_labels = to_categorical(y_train,num_classes=2)
    test_labels = to_categorical(y_test,num_classes=2)

    return X_train, train_labels, X_test, test_labels


# X, y = load_data()
# print(y)