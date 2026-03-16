"""
Vertebral Column dataset loader — CT 2.3: SMOTETomek + Imbalance Ratio thay đổi.

Giống Vertebral_column_KF.py nhưng sử dụng train/test split và áp dụng
SMOTETomek trên tập train trước khi trả về, nhằm cân bằng lớp
(Tổ hợp B / C trong kich_ban.md KB3).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from data.common.change_rate_data import change_rate_data
from data.common.smote_tomek import apply_smote_tomek


def load_data(test_size, new_rate):
    data = pd.read_csv('D:/research/Thu-Nghiem/data/datasets/Vertebral_column.csv')
    diag_map = {'Abnormal': -1.0, 'Normal': 1.0}
    data['Label class'] = data['Label class'].map(diag_map)
    X = data.values[:, 0:-1]
    y = data.values[:, 6]

    X, y = change_rate_data(X, y, new_rate=new_rate)
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, random_state=42, stratify=y)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # CT 2.3: Chỉ áp dụng SMOTETomek trên tập train
    X_train, y_train = apply_smote_tomek(X_train, y_train)

    return X_train, y_train, X_test, y_test
