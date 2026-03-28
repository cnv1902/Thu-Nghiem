"""
KC1 (Software Faults) dataset loader — SMOTETomek.

Giống Kc1_TestSize.py nhưng áp dụng SMOTETomek trên tập train
trước khi trả về, nhằm cân bằng lớp.
"""

from pathlib import Path

from data.common.software_fault_loader import load_software_fault_data


def load_data(test_size, new_rate=0.2):
    dataset_path = Path(__file__).resolve().parent / "datasets" / "Software-Faults" / "kc1.csv"
    return load_software_fault_data(
        path=str(dataset_path),
        test_size=test_size,
        use_smote_tomek=True,
        new_rate=new_rate,
    )
