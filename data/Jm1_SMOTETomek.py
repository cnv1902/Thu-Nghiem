"""
JM1 (Software Faults) dataset loader — CT 2.3: SMOTETomek.

Giống Jm1_TestSize.py nhưng áp dụng SMOTETomek trên tập train
trước khi trả về, nhằm cân bằng lớp (Tổ hợp B / C trong kich_ban.md KB3).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from data.common.smote_tomek import apply_smote_tomek


def load_data(test_size):
    dataset_path = Path(__file__).resolve().parent / "datasets" / "Software-Faults" / "jm1_2000_0.02.csv"
    df = pd.read_csv(
        dataset_path,
        na_values=['?']
    )

    y_raw = df['defects'].astype(str).str.strip().str.upper()
    df['defects'] = y_raw.map({'TRUE': 1, 'FALSE': -1})

    if df['defects'].isna().any():
        bad_vals = y_raw[df['defects'].isna()].unique()
        raise ValueError(f"Invalid label values in defects: {bad_vals[:10]}")

    X = df.drop(columns=['defects'])
    y = df['defects']

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    X = np.log1p(X)

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # CT 2.3: Chỉ áp dụng SMOTETomek trên tập train
    X_train, y_train = apply_smote_tomek(X_train, y_train)

    return X_train, np.asarray(y_train), X_test, np.asarray(y_test)
