import numpy as np
import methods
import svm
from gpu_backend import get_array_module, to_backend_array, cleanup_gpu_memory


def fit(
    X,
    y,
    M=10,
    C=None,
    instance_categorization=False,
    proposed_preprocessing=False,
    theta=1,
    use_entropy_init=False,
    use_fuzzy_spatial_weight=True,
    delta=1e-6,
    knn_k=9,
    lambda_se=1.28,
    mu_se=0.4,
    class_weight=None,
    use_gpu=False,
    cleanup_gpu=False,
):
    """
    FEARN-AdaBoost trainer cho weak learner SVM/WSVM.

    Kich hoat:
    - Soft-margin violation (de xuat 2)
    - FEARN confidence eps* / alpha* (de xuat 3)
    - Fuzzy spatial weight f(x) neu use_fuzzy_spatial_weight=True (de xuat 1)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    N, d = X.shape

    xp, gpu_enabled = get_array_module(use_gpu)
    X_backend = to_backend_array(X, xp) if gpu_enabled else X
    y_backend = to_backend_array(y, xp) if gpu_enabled else y

    if class_weight is None:
        class_weight = {-1: 1.0, 1: lambda_se}

    # A-FEARN pseudocode uses unbiased initialization W_1(i) = 1/N.
    if use_entropy_init:
        W_ada = methods.entropy_init_weight(X, y, proposed=proposed_preprocessing)
    else:
        W_ada = np.ones(N) / N

    fuzzy_weight = methods.compute_fuzzy_spatial_weight(
        X_backend,
        y_backend,
        delta=delta,
        xp=xp,
        k=knn_k,
    )

    w = []
    b = []
    alpha = []

    if instance_categorization:
        B_ada = methods.intinitialization_instance_categorization(N)
    for _ in range(M):
        current_w = W_ada * B_ada if instance_categorization else W_ada
        wi, bi = svm.fit(
            X,
            y,
            C,
            distribution_weight=current_w,
            class_weight=class_weight,
        )

        w.append(wi)
        b.append(bi)

        wi_backend = to_backend_array(wi, xp) if gpu_enabled else wi
        raw_margin = X_backend.dot(wi_backend) + bi
        alpha_i, eps_star, eps_neg, eps_pos, gamma = methods.fearn_confident(
            W_ada,
            y,
            raw_margin,
            fuzzy_weight=fuzzy_weight,
            use_fuzzy_spatial_weight=use_fuzzy_spatial_weight,
            mu_se=mu_se,
            clip_bound=50.0,
            xp=xp,
        )
        alpha.append(alpha_i)

        pred_i = np.sign(X.dot(wi) + bi)
        true_index, false_index, _, _ = methods.find_true_false_index(y, pred_i)
        W_ada = methods.update_weight_adjustment(W_ada, alpha_i, true_index, false_index)

        if instance_categorization:
            B_ada = methods.update_instance_categorization_final(X, y, wi, bi)

        if cleanup_gpu:
            cleanup_gpu_memory()

    return w, b, alpha


def predict(
    X,
    w,
    b,
    alpha,
    M=10,
    delta_threshold=0.0,
    use_gpu=False,
    cleanup_gpu=False,
):
    """
    Ensemble vote dung cong thuc:
    H(x) = sign(sum_t alpha_t * sign(w_t.x + b_t)).
    """
    X = np.asarray(X)
    xp, gpu_enabled = get_array_module(use_gpu)

    if gpu_enabled:
        X_backend = to_backend_array(X, xp)
        H = xp.zeros(X_backend.shape[0])
    else:
        X_backend = X
        H = np.zeros(X.shape[0])
    loops = min(M, len(alpha), len(w), len(b))
    for i in range(loops):
        if gpu_enabled:
            wi_backend = to_backend_array(w[i], xp)
            h_i = xp.sign(X_backend.dot(wi_backend) + b[i])
        else:
            h_i = np.sign(X_backend.dot(w[i]) + b[i])
        H += alpha[i] * h_i

    if gpu_enabled:
        y_pred = np.sign(np.asarray(H.get()) + delta_threshold)
    else:
        y_pred = np.sign(H + delta_threshold)

    if cleanup_gpu:
        cleanup_gpu_memory()
    return y_pred
