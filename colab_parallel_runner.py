"""
Example runner for Colab/CPU-GPU hybrid execution using Joblib loky.

This script demonstrates how to replace ThreadPool-style orchestration
with process-based parallel training tasks that are pickle-safe.
"""

from datetime import datetime
import csv
import os
import math

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
)

from parallel_experiment_engine import run_tasks_parallel


METHOD_REGISTRY = {
    # Traditional Algorithms
    "Decision Tree": "experiment_methods:run_decision_tree",
    "SVM (lib)": "experiment_methods:run_svm",
    "WSVM": "experiment_methods:run_wsvm",
    "ADA_DSTree": "experiment_methods:run_adaboost_decisiontree",
    "ADA_SVM": "experiment_methods:run_adaboost_svm",
    "ADA_WSVM": "experiment_methods:run_adaboost_wsvm",

    # ImADA family
    "ImADA_12_DecisionTree": "experiment_methods:run_imada_12_decisiontree",
    "ImADA_12_SVM": "experiment_methods:run_imada_12_svm",
    "ImADA_12_WSVM": "experiment_methods:run_imada_12_wsvm",
    
    # Improved Algorithms
    "FEARN_AdaBoost_SVM": "experiment_methods:run_fearn_adaboost_svm",
    "FEARN_AdaBoost_WSVM": "experiment_methods:run_fearn_adaboost_wsvm",
}


def build_fearn_tasks(
    X_train,
    y_train,
    X_test,
    y_test,
    M_values,
    C_values,
    theta_values,
    variant_tag,
    test_size,
    dataset_name="unknown",
    fearn_k=9,
    fearn_lambda_se=1.28,
    fearn_mu_se=0.4,
    method_switches=None,
):
    if method_switches is None:
        method_switches = {
            "Decision Tree": False,
            "SVM (lib)": False,
            "WSVM": False,
            "ADA_DSTree": False,
            "ADA_SVM": False,
            "ADA_WSVM": False,
            "ImADA_12_DecisionTree": False,
            "ImADA_12_SVM": False,
            "ImADA_12_WSVM": False,
            "FEARN_AdaBoost_SVM": True,
            "FEARN_AdaBoost_WSVM": False,
        }

    tasks = []
    active_methods = [m for m, enabled in method_switches.items() if enabled and m in METHOD_REGISTRY]
    print(f"Active methods: {active_methods}")

    for method_name, enabled in method_switches.items():
        if not enabled:
            continue
        if method_name not in METHOD_REGISTRY:
            continue

        needs_m = "ADA" in method_name or "AdaBoost" in method_name
        needs_c = "SVM" in method_name or "WSVM" in method_name or "FEARN" in method_name
        
        m_list = M_values if needs_m else [None]
        c_list = C_values if needs_c else [None]

        needs_theta = method_name.startswith("ImADA_12")
        theta_list = theta_values if needs_theta else [None]

        for m in m_list:
            for c in c_list:
                for th in theta_list:
                    method_kwargs = {
                        "M": m,
                        "C": c,
                        "X_train": X_train,
                        "y_train": y_train,
                        "X_test": X_test,
                    }
                    if needs_theta:
                        method_kwargs["theta"] = th
                    if method_name.startswith("FEARN_AdaBoost"):
                        method_kwargs["K"] = fearn_k
                        method_kwargs["lambda_se"] = fearn_lambda_se
                        method_kwargs["mu_se"] = fearn_mu_se

                    tasks.append(
                        {
                            "method_path": METHOD_REGISTRY[method_name],
                            "method_kwargs": method_kwargs,
                            "metadata": {
                                "dataset": dataset_name,
                                "method": method_name,
                                "M": m if m is not None else "N/A",
                                "C": c if c is not None else "N/A",
                                "theta": th if th is not None else "None",
                                "variant": variant_tag,
                                "test_size": test_size,
                                "y_test": y_test,
                            },
                            "cleanup_gpu": True,
                        }
                    )
    return tasks


def _compute_metrics_like_old(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel()

    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    gmean = math.sqrt(sp * se)
    f1s = f1_score(y_test, y_pred, zero_division=0)
    pre = precision_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_pred)
    except Exception:
        auc = "ERR"

    return sp, se, gmean, f1s, pre, acc, auc, cm


def save_raw_parallel_results(results, output_csv_path):
    with open(output_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            md = r["metadata"]

            if r.get("ok"):
                y_test = md.get("y_test")
                y_pred = r.get("y_pred")
                sp, se, gmean, f1s, pre, acc, auc, cm = _compute_metrics_like_old(
                    y_test, y_pred
                )
                writer.writerow(
                    [
                        md.get("dataset", "unknown"),
                        md["test_size"],
                        f"{md['method']} | {md['variant']}",
                        md["M"],
                        md["C"],
                        md["theta"],
                        sp,
                        se,
                        gmean,
                        f1s,
                        pre,
                        acc,
                        auc,
                        str(cm),
                        "None",
                        r.get("alpha"),
                    ]
                )
            else:
                writer.writerow(
                    [
                        md.get("dataset", "unknown"),
                        md.get("test_size"),
                        f"{md.get('method')} | {md.get('variant')}",
                        md.get("M"),
                        md.get("C"),
                        md.get("theta"),
                        "ERR",
                        "ERR",
                        "ERR",
                        "ERR",
                        "ERR",
                        "ERR",
                        "ERR",
                        str(r.get("error")),
                        "None",
                        "None",
                    ]
                )


def init_raw_parallel_results_file(output_csv_path):
    header = [
        "Dataset",
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
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def run_parallel_scenario(
    X_train,
    y_train,
    X_test,
    y_test,
    dataset_name="unknown",
    variant_tag="ORIG",
    test_size=0.2,
    M_values=None,
    C_values=None,
    theta_values=None,
    fearn_k=9,
    fearn_lambda_se=1.28,
    fearn_mu_se=0.4,
    method_switches=None,
    n_jobs=-1,
    batch_size=4,
):
    if M_values is None:
        M_values = [10, 15, 20, 25]
    if C_values is None:
        C_values = [10, 100, 1000]
    if theta_values is None:
        theta_values = [1.0]

    tasks = build_fearn_tasks(
        X_train,
        y_train,
        X_test,
        y_test,
        M_values=M_values,
        C_values=C_values,
        theta_values=theta_values,
        dataset_name=dataset_name,
        fearn_k=fearn_k,
        fearn_lambda_se=fearn_lambda_se,
        fearn_mu_se=fearn_mu_se,
        variant_tag=variant_tag,
        test_size=test_size,
        method_switches=method_switches,
    )

    # Create output file immediately so users can see where results will appear.
    ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    out_dir = "./Experiment"
    os.makedirs(out_dir, exist_ok=True)
    safe_dataset = str(dataset_name).replace(" ", "_")
    out = f"{out_dir}/ParallelRaw_{safe_dataset}_{variant_tag}_{ts}.csv"
    init_raw_parallel_results_file(out)

    print(f"Planned tasks: {len(tasks)}")
    print(f"Output file: {out}")

    all_results = []
    total = len(tasks)
    if batch_size <= 0:
        batch_size = total if total > 0 else 1

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_tasks = tasks[start:end]

        batch_results = run_tasks_parallel(
            batch_tasks,
            n_jobs=n_jobs,
            backend="loky",
            verbose=10,
        )
        save_raw_parallel_results(batch_results, out)
        all_results.extend(batch_results)
        print(f"Saved rows: {len(all_results)}/{total}")

    return out, all_results
