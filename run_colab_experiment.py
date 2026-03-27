"""
One-command entrypoint for Colab/local execution with joblib loky parallelism.

Usage:
    python run_colab_experiment.py
"""

from data import Jm1_TestSize, Jm1_SMOTETomek
from colab_parallel_runner import run_parallel_scenario


def load_data_flexible(dataset_module, testsize, new_rate_val=None):
    if new_rate_val is None:
        return dataset_module.load_data(test_size=testsize)
    try:
        return dataset_module.load_data(test_size=testsize, new_rate=new_rate_val)
    except TypeError:
        return dataset_module.load_data(test_size=testsize)


def main():
    # -----------------------------
    # 1) Cau hinh tham so chay
    # -----------------------------
    M = [10, 15, 20, 25]
    C = [10, 100, 1000]
    theta = [0.3, 0.5, 0.7, 1, 1.5, 2]  # Hien tai khong dung trong FEARN/SoftMargin pipeline
    N = 1
    test_size = [0.2]

    dataset_name = "jm1"
    dataset_original = Jm1_TestSize
    dataset_smote = Jm1_SMOTETomek

    new_rate = None

    # So process song song (-1 = dung tat ca core)
    n_jobs = 1

    # -----------------------------
    # 2) Bat/tat method tai day
    # True = chay, False = bo qua
    # -----------------------------
    METHOD_SWITCHES = {
        # Nhom co ban (khong dung M/C/theta) - engine colab hien tai bo qua
        "Decision Tree": False,
        "SVM (lib)": False,
        "ADA_DSTree": False,
        "ADA_SVM": False,

        # Nhom dung theta - engine colab hien tai bo qua
        "WSVM": False,
        "ADA_WSVM": False,
        "ImADA_12_DecisionTree": False,
        "ImADA_12_SVM": False,
        "ImADA_12_WSVM": False,

        # Nhom KHONG dung theta
        "EANR-AdaBoost_DecisionTree": False,
        "EANR-AdaBoost_SVM": False,
        "EANR-AdaBoost_WSVM": False,
        "SoftMargin_EARN_AdaBoost_SVM": True,
        "SoftMargin_EARN_AdaBoost_WSVM": True,
        "FEARN_AdaBoost_SVM": True,
        "FEARN_AdaBoost_WSVM": True,
    }

    dataset_variants = [
        ("ORIG", dataset_original),
        ("SMOTE", dataset_smote),
    ]

    print(f"Run dataset: {dataset_name}")
    for n in range(N):
        print(f"Lap {n + 1}/{N}")
        for ts in test_size:
            for variant_tag, dataset_module in dataset_variants:
                X_train, y_train, X_test, y_test = load_data_flexible(
                    dataset_module, ts, new_rate_val=new_rate
                )

                output_csv, results = run_parallel_scenario(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    variant_tag=variant_tag,
                    test_size=ts,
                    M_values=M,
                    C_values=C,
                    method_switches=METHOD_SWITCHES,
                    n_jobs=n_jobs,
                )

                ok_count = sum(1 for r in results if r.get("ok"))
                print(
                    f"[{variant_tag}] test_size={ts} | Successful tasks: {ok_count}/{len(results)}"
                )
                print(f"Raw results saved to: {output_csv}")


if __name__ == "__main__":
    main()
