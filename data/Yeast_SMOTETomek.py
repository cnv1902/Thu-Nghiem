"""
Yeast dataset loader — CT 2.3: SMOTETomek.

Giống Yeast_TestSize.py nhưng áp dụng SMOTETomek trên tập train
trước khi trả về, nhằm cân bằng lớp (Tổ hợp B / C trong kich_ban.md KB3).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from data.common.smote_tomek import apply_smote_tomek


def load_data(test_size):
    dataset = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/yeast.csv')
    yeast_map = {'ME2': 1, 'CYT': -1, 'ERL': -1, 'EXC': -1, 'ME1': -1,
                 'ME3': -1, 'MIT': -1, 'NUC': -1, 'POX': -1, 'VAC': -1}
    dataset['name'] = dataset['name'].map(yeast_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, random_state=42, stratify=y)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # CT 2.3: Chỉ áp dụng SMOTETomek trên tập train
    X_train, y_train = apply_smote_tomek(X_train, y_train)

    return X_train, y_train, X_test, y_test
