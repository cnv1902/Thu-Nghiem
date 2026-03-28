import numpy as np
import methods
import svm


def fit(
    X,
    y,
    M=10,
    C=None,
    instance_categorization=False,
    proposed_preprocessing=False,
    test_something=True,
    theta=1,
    use_entropy_init=True,
    use_fuzzy_spatial_weight=True,
    delta=1e-6,
):
    """
    FEARN-AdaBoost trainer cho weak learner SVM/WSVM.

    Kich hoat:
    - Soft-margin violation (de xuat 2)
    - FEARN confidence eps* / alpha* (de xuat 3)
    - Fuzzy spatial weight f(x) neu use_fuzzy_spatial_weight=True (de xuat 1)
    """
    N, d = X.shape

    if use_entropy_init:
        W_ada = methods.entropy_init_weight(X, y, proposed=proposed_preprocessing)
    else:
        W_ada = methods.intinitialization_weight_adjustment(X, y, proposed_preprocessing, theta)

    fuzzy_weight = methods.compute_fuzzy_spatial_weight(X, y, delta=delta)

    w = []
    b = []
    alpha = []
    unit_weight = np.ones(N)

    if instance_categorization:
        B_ada = methods.intinitialization_instance_categorization(N)
        for _ in range(M):
            WC = W_ada * B_ada
            if test_something:
                wi, bi = svm.fit(X, y, C, distribution_weight=unit_weight)
            else:
                wi, bi = svm.fit(X, y, C, distribution_weight=WC)

            w.append(wi)
            b.append(bi)

            raw_margin = y * (X.dot(wi) + bi)
            alpha_i, eps_star, eps_neg, eps_pos, gamma = methods.fearn_confident(
                W_ada,
                y,
                raw_margin,
                fuzzy_weight=fuzzy_weight,
                use_fuzzy_spatial_weight=use_fuzzy_spatial_weight,
            )
            alpha.append(alpha_i)

            pred_i = np.sign(X.dot(wi) + bi)
            true_index, false_index, _, _ = methods.find_true_false_index(y, pred_i)
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i, true_index, false_index)
            B_ada = methods.update_instance_categorization_final(X, y, wi, bi)
    else:
        for _ in range(M):
            if test_something:
                wi, bi = svm.fit(X, y, C, distribution_weight=unit_weight)
            else:
                wi, bi = svm.fit(X, y, C, distribution_weight=W_ada)

            w.append(wi)
            b.append(bi)

            raw_margin = y * (X.dot(wi) + bi)
            alpha_i, eps_star, eps_neg, eps_pos, gamma = methods.fearn_confident(
                W_ada,
                y,
                raw_margin,
                fuzzy_weight=fuzzy_weight,
                use_fuzzy_spatial_weight=use_fuzzy_spatial_weight,
            )
            alpha.append(alpha_i)

            pred_i = np.sign(X.dot(wi) + bi)
            true_index, false_index, _, _ = methods.find_true_false_index(y, pred_i)
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i, true_index, false_index)

    return w, b, alpha


def predict(X, w, b, alpha, M=10):
    """
    Ensemble vote dung cong thuc:
    H(x) = sign(sum_t alpha_t * sign(w_t.x + b_t)).
    """
    loops = min(M, len(alpha), len(w), len(b))
    if loops <= 0:
        return np.sign(np.zeros(X.shape[0]))

    w_stack = np.asarray(w[:loops])
    b_vec = np.asarray(b[:loops])
    alpha_vec = np.asarray(alpha[:loops])

    margins = X.dot(w_stack.T) + b_vec
    weak_votes = np.sign(margins)
    H = weak_votes.dot(alpha_vec)
    return np.sign(H)
