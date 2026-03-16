"""
CT 2.3 — SMOTETomek & SMOTEENN preprocessing module.

SMOTETomek kết hợp:
  - SMOTE: tổng hợp mẫu thiểu số (over-sampling)
  - Tomek Links: xóa cặp mẫu biên giới nhiễu (under-sampling)

SMOTEENN kết hợp:
  - SMOTE: tổng hợp mẫu thiểu số
  - ENN (Edited Nearest Neighbors): xóa mẫu phân loại sai bởi KNN

QUAN TRỌNG: Chỉ áp dụng trên tập train (X_train, y_train),
            KHÔNG áp dụng trên tập test để tránh data leakage.

Tham chiếu: kich_ban.md — KB3 (H7, H8, H9), KB4 (A5, A6)
"""

import numpy as np
from imblearn.combine import SMOTETomek, SMOTEENN


def apply_smote_tomek(X_train, y_train, random_state=42):
    """
    CT 2.3: Áp dụng SMOTETomek trên tập huấn luyện.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        Dữ liệu huấn luyện đã được scale (áp dụng sau StandardScaler).
    y_train : ndarray of shape (n_samples,)
        Nhãn lớp ({-1, 1}) tương ứng.
    random_state : int, default=42
        Seed để tái hiện kết quả.

    Returns
    -------
    X_resampled : ndarray
        Dữ liệu huấn luyện sau khi cân bằng lớp.
    y_resampled : ndarray
        Nhãn lớp sau khi cân bằng lớp.
    """
    smt = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def apply_smote_enn(X_train, y_train, random_state=42):
    """
    CT 2.3 variant: Áp dụng SMOTEENN trên tập huấn luyện.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        Dữ liệu huấn luyện đã được scale.
    y_train : ndarray of shape (n_samples,)
        Nhãn lớp ({-1, 1}) tương ứng.
    random_state : int, default=42
        Seed để tái hiện kết quả.

    Returns
    -------
    X_resampled : ndarray
    y_resampled : ndarray
    """
    smote_enn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
