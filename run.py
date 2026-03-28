import os

# Prevent CPU over-subscription when running many models in parallel.
for _env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.svm import SVC

import ImAda_DecisionTree
import fearn_adaboost as fearn_toa
import trainning_of_adaboost as toa
from data import (
    Cm1_SMOTETomek,
    Cm1_TestSize,
    Jm1_SMOTETomek,
    Jm1_TestSize,
    Kc1_SMOTETomek,
    Kc1_TestSize,
    Kc2_SMOTETomek,
    Kc2_TestSize,
    Pc1_SMOTETomek,
    Pc1_TestSize,
    Pc2_SMOTETomek,
    Pc2_TestSize,
)
from wsvm.application import Wsvm


def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float, float, float, np.ndarray]:
    cm = confusion_matrix(y_test, y_pred)
    se = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    gmean = math.sqrt(se * sp)
    f1s = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred)
    return sp, se, gmean, f1s, pre, acc, auc, cm


# -----------------------------
# Model wrappers (same logic as notebook)
# -----------------------------
def svm_lib(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    clf = SVC(probability=True, kernel="linear")
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def decisiontree(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def wsvm(C: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, distribution_weight: Optional[np.ndarray] = None) -> np.ndarray:
    model = Wsvm(C, distribution_weight)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def ada_svm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    clf = AdaBoostClassifier(SVC(probability=True, kernel="linear"), n_estimators=100, learning_rate=1.0)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def ada_decisiontree(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def ada_wsvm(M: int, C: float, theta: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    w, b, a = toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=True,
        proposed_preprocessing=False,
        proposed_alpha=False,
        test_something=False,
        theta=theta,
    )
    y_pred = toa.predict(X_test, w, b, a, M)
    return y_pred, a


def imada_12_wsvm(M: int, C: float, theta: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    w, b, a = toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=True,
        proposed_preprocessing=True,
        proposed_alpha=True,
        test_something=False,
        theta=theta,
    )
    y_pred = toa.predict(X_test, w, b, a, M)
    return y_pred, a


def imada_12_svm(M: int, C: float, theta: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    w, b, a = toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=False,
        proposed_preprocessing=True,
        proposed_alpha=True,
        test_something=False,
        theta=theta,
    )
    y_pred = toa.predict(X_test, w, b, a, M)
    return y_pred, a


def imada_12_decisiontree(M: int, theta: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    clfs, a = ImAda_DecisionTree.fit(
        X_train,
        y_train,
        M,
        proposed_preprocessing=True,
        proposed_alpha=True,
        theta=theta,
    )
    y_pred = ImAda_DecisionTree.predict(X_test, a, clfs)
    return y_pred, a


def eanr_adaboost_wsvm(M: int, C: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    w, b, a = toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=True,
        proposed_preprocessing=True,
        proposed_alpha=True,
        test_something=False,
        use_entropy_init=True,
        use_noise_robust_confident=True,
    )
    y_pred = toa.predict(X_test, w, b, a, M)
    return y_pred, a


def eanr_adaboost_svm(M: int, C: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    w, b, a = toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=False,
        proposed_preprocessing=True,
        proposed_alpha=True,
        test_something=False,
        use_entropy_init=True,
        use_noise_robust_confident=True,
    )
    y_pred = toa.predict(X_test, w, b, a, M)
    return y_pred, a


def eanr_adaboost_decisiontree(M: int, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    clfs, a = ImAda_DecisionTree.fit(
        X_train,
        y_train,
        M,
        proposed_preprocessing=True,
        proposed_alpha=True,
        use_entropy_init=True,
        use_noise_robust_confident=True,
    )
    y_pred = ImAda_DecisionTree.predict(X_test, a, clfs)
    return y_pred, a


def softmargin_earn_adaboost_wsvm(M: int, C: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    w, b, a = fearn_toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=True,
        proposed_preprocessing=True,
        test_something=False,
        use_entropy_init=True,
        use_fuzzy_spatial_weight=False,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


def softmargin_earn_adaboost_svm(M: int, C: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    w, b, a = fearn_toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=False,
        proposed_preprocessing=True,
        test_something=False,
        use_entropy_init=True,
        use_fuzzy_spatial_weight=False,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


def fearn_adaboost_wsvm(M: int, C: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    w, b, a = fearn_toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=True,
        proposed_preprocessing=True,
        test_something=False,
        use_entropy_init=True,
        use_fuzzy_spatial_weight=True,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


def fearn_adaboost_svm(M: int, C: float, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    w, b, a = fearn_toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=False,
        proposed_preprocessing=True,
        test_something=False,
        use_entropy_init=True,
        use_fuzzy_spatial_weight=True,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


# -----------------------------
# Configuration and task building
# -----------------------------
HEADER = [
    "Test Size",
    "Method",
    "M",
    "C",
    "theta",
    "SP",
    "SE",
    "Gmean",
    "F1 Score",
    "Precision",
    "Accuracy",
    "AUC",
    "Ma tran nham lan",
    "List of err_w",
    "List of alpha",
]

DEFAULT_METHOD_SWITCHES: Dict[str, bool] = {
    # Nhom co ban (khong dung M/C/theta)
    "Decision Tree": True,
    "SVM (lib)": True,
    "ADA_DSTree": True,
    "ADA_SVM": True,
    # Nhom dung theta
    "WSVM": True,
    "ADA_WSVM": True,
    "ImADA_12_DecisionTree": True,
    "ImADA_12_SVM": True,
    "ImADA_12_WSVM": True,
    # Nhom KHONG dung theta
    "EANR-AdaBoost_DecisionTree": True,
    "EANR-AdaBoost_SVM": True,
    "EANR-AdaBoost_WSVM": True,
    "SoftMargin_EARN_AdaBoost_SVM": False,
    "SoftMargin_EARN_AdaBoost_WSVM": False,
    "FEARN_AdaBoost_SVM": False,
    "FEARN_AdaBoost_WSVM": False,
}

DATASET_REGISTRY = {
    "jm1": (Jm1_TestSize, Jm1_SMOTETomek),
    "cm1": (Cm1_TestSize, Cm1_SMOTETomek),
    "kc1": (Kc1_TestSize, Kc1_SMOTETomek),
    "kc2": (Kc2_TestSize, Kc2_SMOTETomek),
    "pc1": (Pc1_TestSize, Pc1_SMOTETomek),
    "pc2": (Pc2_TestSize, Pc2_SMOTETomek),
}

# cp2 alias for compatibility with user shorthand.
DATASET_ALIAS = {
    "cp2": "pc2",
}

DATASET_SWITCHES: Dict[str, bool] = {
    "jm1": True,
    "cm1": False,
    "kc1": False,
    "kc2": False,
    "pc1": False,
    "pc2": False,
}


@dataclass(frozen=True)
class TaskSpec:
    method_key: str
    variant_tag: str
    testsize: float
    m: object
    c: object
    t: object


def load_data_flexible(dataset_module, testsize: float, new_rate_val: Optional[float] = None):
    if new_rate_val is None:
        return dataset_module.load_data(test_size=testsize)
    try:
        return dataset_module.load_data(test_size=testsize, new_rate=new_rate_val)
    except TypeError:
        return dataset_module.load_data(test_size=testsize)


def _legacy_vote(X: np.ndarray, w: Sequence[np.ndarray], b: Sequence[float], alpha: Sequence[float], M: int) -> np.ndarray:
    H = np.zeros(X.shape[0])
    loops = min(M, len(alpha), len(w), len(b))
    for i in range(loops):
        H += alpha[i] * np.sign(X.dot(w[i]) + b[i])
    return np.sign(H)


def _vectorized_vote(X: np.ndarray, w: Sequence[np.ndarray], b: Sequence[float], alpha: Sequence[float], M: int) -> np.ndarray:
    loops = min(M, len(alpha), len(w), len(b))
    if loops <= 0:
        return np.sign(np.zeros(X.shape[0]))
    w_stack = np.asarray(w[:loops])
    b_vec = np.asarray(b[:loops])
    alpha_vec = np.asarray(alpha[:loops])
    return np.sign(np.sign(X.dot(w_stack.T) + b_vec).dot(alpha_vec))


def verify_vectorized_equivalence() -> None:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((64, 12))
    w = [rng.standard_normal(12) for _ in range(7)]
    b = [float(v) for v in rng.standard_normal(7)]
    alpha = [float(v) for v in rng.standard_normal(7)]

    legacy = _legacy_vote(X, w, b, alpha, M=7)
    vectorized = _vectorized_vote(X, w, b, alpha, M=7)

    assert np.array_equal(legacy, vectorized), "Vectorized ensemble vote must match legacy loop vote"


def build_task_specs(method_switches: Dict[str, bool], variant_tag: str, testsize: float, M_values: Sequence[int], C_values: Sequence[float], theta_values: Sequence[float]) -> List[TaskSpec]:
    specs: List[TaskSpec] = []

    def add(method_key: str, m: object, c: object, t: object) -> None:
        if method_switches.get(method_key, False):
            specs.append(TaskSpec(method_key=method_key, variant_tag=variant_tag, testsize=testsize, m=m, c=c, t=t))

    add("Decision Tree", "None", "none", "none")
    add("SVM (lib)", "None", "none", "none")
    add("ADA_DSTree", "None", "none", "none")
    add("ADA_SVM", "None", "none", "none")

    for m in M_values:
        for c in C_values:
            # Keep notebook behavior: theta-based group remains disabled by default.
            for t in theta_values:
                add("WSVM", m, c, t)
                add("ADA_WSVM", m, c, t)
                add("ImADA_12_DecisionTree", m, c, t)
                add("ImADA_12_SVM", m, c, t)
                add("ImADA_12_WSVM", m, c, t)

            add("EANR-AdaBoost_DecisionTree", m, c, "None")
            add("EANR-AdaBoost_SVM", m, c, "None")
            add("EANR-AdaBoost_WSVM", m, c, "None")
            add("SoftMargin_EARN_AdaBoost_SVM", m, c, "None")
            add("SoftMargin_EARN_AdaBoost_WSVM", m, c, "None")
            add("FEARN_AdaBoost_SVM", m, c, "None")
            add("FEARN_AdaBoost_WSVM", m, c, "None")

    return specs


def run_one_task(
    spec: TaskSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    distribution_weight: np.ndarray,
) -> List[object]:
    method_name = f"{spec.method_key} | {spec.variant_tag}"

    try:
        if spec.method_key == "Decision Tree":
            y_pred, alpha = decisiontree(X_train, y_train, X_test), "None"
        elif spec.method_key == "SVM (lib)":
            y_pred, alpha = svm_lib(X_train, y_train, X_test), "None"
        elif spec.method_key == "ADA_DSTree":
            y_pred, alpha = ada_decisiontree(X_train, y_train, X_test), "None"
        elif spec.method_key == "ADA_SVM":
            y_pred, alpha = ada_svm(X_train, y_train, X_test), "None"
        elif spec.method_key == "WSVM":
            y_pred, alpha = wsvm(spec.c, X_train, y_train, X_test, distribution_weight), "None"
        elif spec.method_key == "ADA_WSVM":
            y_pred, alpha = ada_wsvm(spec.m, spec.c, spec.t, X_train, y_train, X_test)
        elif spec.method_key == "ImADA_12_DecisionTree":
            y_pred, alpha = imada_12_decisiontree(spec.m, spec.t, X_train, y_train, X_test)
        elif spec.method_key == "ImADA_12_SVM":
            y_pred, alpha = imada_12_svm(spec.m, spec.c, spec.t, X_train, y_train, X_test)
        elif spec.method_key == "ImADA_12_WSVM":
            y_pred, alpha = imada_12_wsvm(spec.m, spec.c, spec.t, X_train, y_train, X_test)
        elif spec.method_key == "EANR-AdaBoost_DecisionTree":
            y_pred, alpha = eanr_adaboost_decisiontree(spec.m, X_train, y_train, X_test)
        elif spec.method_key == "EANR-AdaBoost_SVM":
            y_pred, alpha = eanr_adaboost_svm(spec.m, spec.c, X_train, y_train, X_test)
        elif spec.method_key == "EANR-AdaBoost_WSVM":
            y_pred, alpha = eanr_adaboost_wsvm(spec.m, spec.c, X_train, y_train, X_test)
        elif spec.method_key == "SoftMargin_EARN_AdaBoost_SVM":
            y_pred, alpha = softmargin_earn_adaboost_svm(spec.m, spec.c, X_train, y_train, X_test)
        elif spec.method_key == "SoftMargin_EARN_AdaBoost_WSVM":
            y_pred, alpha = softmargin_earn_adaboost_wsvm(spec.m, spec.c, X_train, y_train, X_test)
        elif spec.method_key == "FEARN_AdaBoost_SVM":
            y_pred, alpha = fearn_adaboost_svm(spec.m, spec.c, X_train, y_train, X_test)
        elif spec.method_key == "FEARN_AdaBoost_WSVM":
            y_pred, alpha = fearn_adaboost_wsvm(spec.m, spec.c, X_train, y_train, X_test)
        else:
            raise ValueError(f"Unknown method key: {spec.method_key}")

        sp, se, gmean, f1s, pre, acc, auc, cm = compute_metrics(y_test, y_pred)
        return [
            spec.testsize,
            method_name,
            spec.m,
            spec.c,
            spec.t,
            sp,
            se,
            gmean,
            f1s,
            pre,
            acc,
            auc,
            str(cm),
            "None",
            alpha,
        ]
    except Exception as ex:
        return [
            spec.testsize,
            method_name,
            spec.m,
            spec.c,
            spec.t,
            "ERR",
            "ERR",
            "ERR",
            "ERR",
            "ERR",
            "ERR",
            "ERR",
            str(ex),
            "None",
            "None",
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ImAda/FEARN experiments as a standalone Python script.")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit dataset list. Supported: jm1 cm1 kc1 kc2 pc1 pc2 cp2",
    )
    parser.add_argument("--trials", type=int, default=1, help="Number of repeat runs (N).")
    parser.add_argument("--test-size", type=float, nargs="+", default=[0.2], dest="test_sizes", help="List of test_size values.")
    parser.add_argument("--m-values", type=int, nargs="+", default=[10, 15, 20, 25], dest="m_values", help="List of M values.")
    parser.add_argument("--c-values", type=float, nargs="+", default=[10, 100, 1000], dest="c_values", help="List of C values.")
    parser.add_argument("--theta-values", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1, 1.5, 2], dest="theta_values", help="List of theta values.")
    parser.add_argument("--new-rate", type=float, default=None, help="Optional class-rate adjustment (for datasets supporting new_rate).")
    parser.add_argument("--max-workers", type=int, default=min(32, (os.cpu_count() or 20)), help="Parallel workers used by joblib.")
    parser.add_argument(
        "--joblib-verbose",
        type=int,
        default=10,
        help="Verbosity level for joblib progress logs (0 = silent).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./Experiment",
        help="Output directory. The script always writes one CSV per dataset.",
    )
    parser.add_argument("--verify", action="store_true", help="Run assertion to verify loop and vectorized vote equivalence before training.")
    return parser.parse_args()


def normalize_dataset_name(name: str) -> str:
    lower_name = name.strip().lower()
    return DATASET_ALIAS.get(lower_name, lower_name)


def selected_datasets(explicit_names: Optional[Sequence[str]]) -> List[str]:
    if explicit_names:
        normalized = [normalize_dataset_name(name) for name in explicit_names]
        invalid = [name for name in normalized if name not in DATASET_REGISTRY]
        if invalid:
            raise ValueError(f"Unsupported dataset(s): {invalid}. Supported: {sorted(DATASET_REGISTRY.keys()) + sorted(DATASET_ALIAS.keys())}")
        # Keep order provided by user while removing duplicates.
        ordered_unique = list(dict.fromkeys(normalized))
        return ordered_unique

    enabled = [name for name, enabled_flag in DATASET_SWITCHES.items() if enabled_flag]
    if not enabled:
        raise ValueError("No dataset is enabled. Set at least one dataset to True in DATASET_SWITCHES or pass --datasets.")
    return enabled


def main() -> None:
    args = parse_args()

    if args.verify:
        verify_vectorized_equivalence()

    active_datasets = selected_datasets(args.datasets)
    time_tag = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = args.output_dir

    output_paths: List[str] = []

    for dataset_name in active_datasets:
        dataset_original, dataset_smote = DATASET_REGISTRY[dataset_name]
        dataset_variants = [("ORIG", dataset_original), ("SMOTE", dataset_smote)]

        output_path = os.path.join(output_dir, f"Data_{dataset_name}_{time_tag}_TestSize.csv")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

        for n in range(args.trials):
            print(f"Lan boc: {n + 1} | Dataset: {dataset_name}", flush=True)
            for testsize in args.test_sizes:
                for variant_tag, dataset_module in dataset_variants:
                    print(f"Dataset variant: {variant_tag}", flush=True)
                    X_train, y_train, X_test, y_test = load_data_flexible(dataset_module, testsize, new_rate_val=args.new_rate)

                    y_train = np.asarray(y_train)
                    y_test = np.asarray(y_test)

                    distribution_weight = np.ones(X_train.shape[0])

                    task_specs = build_task_specs(
                        method_switches=DEFAULT_METHOD_SWITCHES,
                        variant_tag=variant_tag,
                        testsize=testsize,
                        M_values=args.m_values,
                        C_values=args.c_values,
                        theta_values=args.theta_values,
                    )

                    if not task_specs:
                        continue

                    print(
                        f"Start batch: dataset={dataset_name}, variant={variant_tag}, test_size={testsize}, "
                        f"tasks={len(task_specs)}, max_workers={args.max_workers}",
                        flush=True,
                    )

                    rows = Parallel(
                        n_jobs=args.max_workers,
                        prefer="threads",
                        batch_size=1,
                        verbose=args.joblib_verbose,
                    )(
                        delayed(run_one_task)(spec, X_train, y_train, X_test, y_test, distribution_weight) for spec in task_specs
                    )

                    print(
                        f"Finished batch: dataset={dataset_name}, variant={variant_tag}, rows={len(rows)}",
                        flush=True,
                    )

                    with open(output_path, "a", encoding="UTF8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)

        output_paths.append(output_path)

    if len(output_paths) == 1:
        print(f"Hoan tat. Da luu ket qua tai: {output_paths[0]}")
    else:
        print(f"Hoan tat. Da luu ket qua cho {len(output_paths)} datasets tai thu muc: {output_dir}")


if __name__ == "__main__":
    main()
