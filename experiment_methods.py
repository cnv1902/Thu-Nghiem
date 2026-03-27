import fearn_adaboost as fearn_toa
import trainning_of_adaboost as toa
import ImAda_DecisionTree


def run_softmargin_earn_adaboost_wsvm(M, C, X_train, y_train, X_test):
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
        cleanup_gpu=True,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


def run_softmargin_earn_adaboost_svm(M, C, X_train, y_train, X_test):
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
        cleanup_gpu=True,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


def run_fearn_adaboost_wsvm(M, C, X_train, y_train, X_test):
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
        cleanup_gpu=True,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


def run_fearn_adaboost_svm(M, C, X_train, y_train, X_test):
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
        cleanup_gpu=True,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


def run_eanr_adaboost_wsvm(M, C, X_train, y_train, X_test):
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
        cleanup_gpu=True,
    )
    y_pred = toa.predict(X_test, w, b, a, M, cleanup_gpu=True)
    return y_pred, a


def run_eanr_adaboost_svm(M, C, X_train, y_train, X_test):
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
        cleanup_gpu=True,
    )
    y_pred = toa.predict(X_test, w, b, a, M, cleanup_gpu=True)
    return y_pred, a


def run_eanr_adaboost_decisiontree(M, X_train, y_train, X_test):
    clf, a = ImAda_DecisionTree.fit(
        X_train,
        y_train,
        M,
        proposed_preprocessing=True,
        proposed_alpha=True,
        use_entropy_init=True,
        use_noise_robust_confident=True,
    )
    y_pred = ImAda_DecisionTree.predict(X_test, a, clf)
    return y_pred, a
