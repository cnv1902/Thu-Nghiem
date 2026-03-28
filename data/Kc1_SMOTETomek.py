"""
KC1 (Software Faults) dataset loader — SMOTETomek.

Giống Kc1_TestSize.py nhưng áp dụng SMOTETomek trên tập train
trước khi trả về, nhằm cân bằng lớp.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from data.common.change_rate_data import change_rate_data
from data.common.smote_tomek import apply_smote_tomek


def load_data(test_size, new_rate=0.2):
    dataset_path = Path(__file__).resolve().parent / "datasets" / "Software-Faults" / "jm1_500_0.25.csv"
    df = pd.read_csv(
        dataset_path,
        na_values=['?']
    )

    y_raw = df['defects'].astype(str).str.strip().str.upper()
    df['defects'] = y_raw.map({'TRUE': 1, 'FALSE': -1})

    if df['defects'].isna().any():
        bad_vals = y_raw[df['defects'].isna()].unique()
        raise ValueError(f"Invalid label values in defects: {bad_vals[:10]}")

    X = df.drop(columns=['defects']).apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    X = np.log1p(X).to_numpy()
    y = df['defects'].to_numpy()

    X, y = change_rate_data(X, y, new_rate=new_rate)

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = apply_smote_tomek(X_train, y_train)

    return X_train, np.asarray(y_train), X_test, np.asarray(y_test)
