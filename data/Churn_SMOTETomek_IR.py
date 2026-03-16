"""
Churn dataset loader — CT 2.3: SMOTETomek + Imbalance Ratio thay đổi.

Giống churn.py nhưng áp dụng SMOTETomek trên tập train
trước khi trả về, nhằm cân bằng lớp (Tổ hợp B / C trong kich_ban.md KB3).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data.common.change_rate_data import change_rate_data
from data.common.smote_tomek import apply_smote_tomek


def load_data(test_size, new_rate):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/churn.csv')
    Churn_map = {'False.': -1, 'True.': 1}
    dataset['Churn?'] = dataset['Churn?'].map(Churn_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 20].values

    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])   # State
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])   # Area Code
    X[:, 4] = labelencoder_X.fit_transform(X[:, 4])   # Phone
    X[:, 5] = labelencoder_X.fit_transform(X[:, 5])   # Int'l Plan
    X[:, 6] = labelencoder_X.fit_transform(X[:, 6])   # VMail Plan

    X, y = change_rate_data(X, y, new_rate=new_rate)
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, random_state=42, stratify=y)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    pca = PCA(n_components=15)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # CT 2.3: Chỉ áp dụng SMOTETomek trên tập train
    X_train, y_train = apply_smote_tomek(X_train, y_train)

    return X_train, y_train, X_test, y_test
