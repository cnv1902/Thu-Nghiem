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
    "SoftMargin_EARN_AdaBoost_SVM": "experiment_methods:run_softmargin_earn_adaboost_svm",
    "SoftMargin_EARN_AdaBoost_WSVM": "experiment_methods:run_softmargin_earn_adaboost_wsvm",
    "FEARN_AdaBoost_SVM": "experiment_methods:run_fearn_adaboost_svm",
    "FEARN_AdaBoost_WSVM": "experiment_methods:run_fearn_adaboost_wsvm",
    "EANR-AdaBoost_SVM": "experiment_methods:run_eanr_adaboost_svm",
    "EANR-AdaBoost_WSVM": "experiment_methods:run_eanr_adaboost_wsvm",
}


def build_fearn_tasks(
    X_train,
    y_train,
    X_test,
    y_test,
    M_values,
    C_values,
    variant_tag,
    test_size,
    method_switches=None,
):
    if method_switches is None:
        method_switches = {
            "SoftMargin_EARN_AdaBoost_SVM": True,
            "SoftMargin_EARN_AdaBoost_WSVM": True,
            "FEARN_AdaBoost_SVM": True,
            "FEARN_AdaBoost_WSVM": True,
            "EANR-AdaBoost_SVM": False,
            "EANR-AdaBoost_WSVM": False,
        }

    tasks = []
    for m in M_values:
        for c in C_values:
            for method_name, enabled in method_switches.items():
                if not enabled:
                    continue
                if method_name not in METHOD_REGISTRY:
                    continue

                tasks.append(
                    {
                        "method_path": METHOD_REGISTRY[method_name],
                        "method_kwargs": {
                            "M": m,
                            "C": c,
                            "X_train": X_train,
                            "y_train": y_train,
                            "X_test": X_test,
                        },
                        "metadata": {
                            "method": method_name,
                            "M": m,
                            "C": c,
                            "theta": "None",
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
    variant_tag="ORIG",
    test_size=0.2,
    M_values=None,
    C_values=None,
    method_switches=None,
    n_jobs=-1,
    batch_size=4,
):
    if M_values is None:
        M_values = [10, 15, 20, 25]
    if C_values is None:
        C_values = [10, 100, 1000]

    tasks = build_fearn_tasks(
        X_train,
        y_train,
        X_test,
        y_test,
        M_values=M_values,
        C_values=C_values,
        variant_tag=variant_tag,
        test_size=test_size,
        method_switches=method_switches,
    )

    # Create output file immediately so users can see where results will appear.
    ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    out_dir = "./Experiment"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/ParallelRaw_{variant_tag}_{ts}.csv"
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
