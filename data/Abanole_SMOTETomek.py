"""
Abalone dataset loader — CT 2.3: SMOTETomek.

Giống Abanole_TestSize.py nhưng áp dụng SMOTETomek trên tập train
trước khi trả về, nhằm cân bằng lớp (Tổ hợp B / C trong kich_ban.md KB3).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from data.common.smote_tomek import apply_smote_tomek


def load_data(test_size):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/abalone.csv')
    abalone_map = {
        15: 1,
        1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 10: -1,
        11: -1, 12: -1, 13: -1, 14: -1, 16: -1, 17: -1, 18: -1, 19: -1, 20: -1,
        21: -1, 22: -1, 23: -1, 24: -1, 25: -1, 26: -1, 27: -1, 28: -1, 29: -1
    }
    dataset['Rings'] = dataset['Rings'].map(abalone_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])   # Sex

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, random_state=42, stratify=y)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # CT 2.3: Chỉ áp dụng SMOTETomek trên tập train
    X_train, y_train = apply_smote_tomek(X_train, y_train)

    return X_train, y_train, X_test, y_test
