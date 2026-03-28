import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from data.common.smote_tomek import apply_smote_tomek
from data.common.promise_arff import read_promise_arff_dataframe

def load_data(test_size):
    df = read_promise_arff_dataframe(
        'D:/research/Thu-Nghiem/data/datasets/Software-Faults/pc1.csv'
    )
    label_col = df.columns[-1]
    y_raw = df[label_col].astype(str).str.strip().str.upper()
    df[label_col] = y_raw.map({'TRUE': 1, 'FALSE': -1, 'YES': 1, 'NO': -1})
    if df[label_col].isna().any():
        bad_vals = y_raw[df[label_col].isna()].unique()
        raise ValueError(f"Invalid label values in defects: {bad_vals[:10]}")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    X = np.log1p(X)
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=test_size, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train, y_train = apply_smote_tomek(X_train, y_train)
    return X_train, np.asarray(y_train), X_test, np.asarray(y_test)
