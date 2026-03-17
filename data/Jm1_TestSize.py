import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

def load_data(test_size):

    df = pd.read_csv(
        'D:/research/Thu-Nghiem/data/datasets/Software-Faults/jm2.csv',
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
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train.to_numpy(), X_test, y_test.to_numpy()