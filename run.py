"""
One-command entrypoint for Colab/local execution with joblib loky parallelism.

Usage:
    python run_colab_experiment.py
"""

from data import (
    Cm1_TestSize, Cm1_SMOTETomek,
    Kc1_TestSize, Kc1_SMOTETomek,
    Jm1_TestSize, Jm1_SMOTETomek,
    Kc2_TestSize, Kc2_SMOTETomek,
    Pc1_TestSize, Pc1_SMOTETomek,
    Pc2_TestSize, Pc2_SMOTETomek,
)
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
    C = [10, 100]
    theta = [0.3, 0.5, 0.7, 1, 1.5, 2]  # Duoc dung cho cac method ImADA_12_*
    fearn_k = 9
    fearn_lambda_se = 1.28
    fearn_mu_se = 0.2
    N = 1
    test_size = [0.2]

    # Dataset declarations
    DATASET_SWITCHES = {
        'cm1': True,
        'kc1': True,
        'jm1': True,
        'kc2': True,
        'pc1': True,
        'pc2': True,
    }

    DATASET_VARIANTS = {
        'cm1': [
            ("ORIG", Cm1_TestSize),
            ("SMOTE", Cm1_SMOTETomek),
        ],
        'kc1': [
            ("ORIG", Kc1_TestSize),
            ("SMOTE", Kc1_SMOTETomek),
        ],
        'jm1': [
            ("ORIG", Jm1_TestSize),
            ("SMOTE", Jm1_SMOTETomek),
        ],
        'kc2': [
            ("ORIG", Kc2_TestSize),
            ("SMOTE", Kc2_SMOTETomek),
        ],
        'pc1': [
            ("ORIG", Pc1_TestSize),
            ("SMOTE", Pc1_SMOTETomek),
        ],
        'pc2': [
            ("ORIG", Pc2_TestSize),
            ("SMOTE", Pc2_SMOTETomek),
        ],
    }

    new_rate = None

    # So process song song (-1 = dung tat ca core)
    n_jobs = 22

    # -----------------------------
    # 2) Bat/tat method tai day
    # True = chay, False = bo qua
    # -----------------------------
    METHOD_SWITCHES = {
        # Nhom truyen thong
        "Decision Tree": True,
        "SVM (lib)": True,
        "WSVM": True,
        "ADA_DSTree": False,
        "ADA_SVM": False,
        "ADA_WSVM": False,

        # Nhom cai tien
        "ImADA_12_DecisionTree": False,
        "ImADA_12_SVM": False,
        "ImADA_12_WSVM": False,
        "FEARN_AdaBoost_SVM": True,
        "FEARN_AdaBoost_WSVM": True,
    }

    for dataset_name, enabled in DATASET_SWITCHES.items():
        if not enabled:
            continue
        dataset_variants = DATASET_VARIANTS[dataset_name]
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
                        dataset_name=dataset_name,
                        variant_tag=variant_tag,
                        test_size=ts,
                        M_values=M,
                        C_values=C,
                        theta_values=theta,
                        fearn_k=fearn_k,
                        fearn_lambda_se=fearn_lambda_se,
                        fearn_mu_se=fearn_mu_se,
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
