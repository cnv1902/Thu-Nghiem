import io
from typing import Tuple

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler

from data.common.smote_tomek import apply_smote_tomek


_LABEL_MAP = {
    "true": 1,
    "false": -1,
    "yes": 1,
    "no": -1,
    "1": 1,
    "-1": -1,
    "1.0": 1,
    "-1.0": -1,
}


def _read_arff_like_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    attr_names = []
    data_start = None

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue

        lower_line = line.lower()
        if lower_line.startswith("@attribute"):
            parts = line.split()
            if len(parts) >= 2:
                attr_names.append(parts[1].strip("'\""))
        elif lower_line.startswith("@data"):
            data_start = idx + 1
            break

    if data_start is None:
        raise ValueError(f"Cannot find @data section in ARFF-like file: {path}")

    data_rows = []
    for raw_line in lines[data_start:]:
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        data_rows.append(line)

    if not data_rows:
        raise ValueError(f"No data rows found after @data section in file: {path}")

    df = pd.read_csv(io.StringIO("\n".join(data_rows)), header=None)

    if attr_names and len(attr_names) == df.shape[1]:
        df.columns = attr_names

    return df


def _read_software_fault_dataframe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, na_values=["?"], comment="%")
        df.columns = df.columns.str.strip()
        lower_cols = {col.lower() for col in df.columns}
        if "defects" in lower_cols or "problems" in lower_cols:
            return df
    except ParserError:
        pass

    df = _read_arff_like_csv(path)
    df.columns = pd.Index([str(col).strip() for col in df.columns])
    return df


def _extract_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    lower_to_col = {str(col).strip().lower(): col for col in df.columns}

    if "defects" in lower_to_col:
        y_col = lower_to_col["defects"]
    elif "problems" in lower_to_col:
        y_col = lower_to_col["problems"]
    else:
        y_col = df.columns[-1]

    y_raw = df[y_col].astype(str).str.strip().str.lower()
    y = y_raw.map(_LABEL_MAP)

    if y.isna().any():
        bad_vals = y_raw[y.isna()].unique()
        raise ValueError(f"Invalid label values in {y_col}: {bad_vals[:10]}")

    X = df.drop(columns=[y_col]).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    X = np.log1p(X).to_numpy()

    return X, y.to_numpy()


def load_software_fault_data(path: str, test_size: float, use_smote_tomek: bool = False):
    df = _read_software_fault_dataframe(path)
    X, y = _extract_xy(df)

    X_train, X_test, y_train, y_test = tts(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if use_smote_tomek:
        X_train, y_train = apply_smote_tomek(X_train, y_train)

    return X_train, np.asarray(y_train), X_test, np.asarray(y_test)
