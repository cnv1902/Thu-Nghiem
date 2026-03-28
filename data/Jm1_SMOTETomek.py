"""
JM1 (Software Faults) dataset loader — CT 2.3: SMOTETomek.

Giống Jm1_TestSize.py nhưng áp dụng SMOTETomek trên tập train
trước khi trả về, nhằm cân bằng lớp (Tổ hợp B / C trong kich_ban.md KB3).
"""

from pathlib import Path

from data.common.software_fault_loader import load_software_fault_data


def load_data(test_size):
    dataset_path = Path(__file__).resolve().parent / "datasets" / "Software-Faults" / "jm1.csv"
    return load_software_fault_data(
        path=str(dataset_path),
        test_size=test_size,
        use_smote_tomek=True,
        new_rate=None,
    )
