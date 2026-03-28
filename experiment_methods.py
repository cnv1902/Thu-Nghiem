import numpy as np
import fearn_adaboost as fearn_toa
import old.trainning_of_adaboost as toa
import ImAda_DecisionTree

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from wsvm.application import Wsvm


def run_decision_tree(X_train, y_train, X_test, **kwargs):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict(X_test), "None"


def run_svm(C, X_train, y_train, X_test, **kwargs):
    c_val = C if C is not None else 1.0
    clf = SVC(kernel="linear", C=c_val, random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict(X_test), "None"


def run_wsvm(C, X_train, y_train, X_test, **kwargs):
    c_val = C if C is not None else 1.0
    N = X_train.shape[0]
    distribution_weight = np.ones(N) / N
    clf = Wsvm(C=c_val, distribution_weight=distribution_weight)
    clf.fit(X_train, y_train)
    return clf.predict(X_test), "None"


def run_adaboost_decisiontree(M, X_train, y_train, X_test, **kwargs):
    m_val = M if M is not None else 50
    estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf = AdaBoostClassifier(estimator=estimator, n_estimators=m_val, algorithm="SAMME", random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict(X_test), "None"


def run_adaboost_svm(M, C, X_train, y_train, X_test, **kwargs):
    m_val = M if M is not None else 50
    c_val = C if C is not None else 1.0
    estimator = SVC(kernel="linear", C=c_val, probability=True, random_state=42)
    clf = AdaBoostClassifier(estimator=estimator, n_estimators=m_val, algorithm="SAMME", random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict(X_test), "None"


def run_adaboost_wsvm(M, C, X_train, y_train, X_test, **kwargs):
    m_val = M if M is not None else 50
    c_val = C if C is not None else 1.0
    N = X_train.shape[0]
    w = np.ones(N) / N
    
    estimators = []
    estimator_weights = []
    
    for _ in range(m_val):
        clf = Wsvm(C=c_val, distribution_weight=w.copy())
        clf.fit(X_train, y_train)
        pred = clf.predict(X_train)
        
        err = np.sum(w[pred != y_train]) / np.sum(w)
        if err >= 0.5:
            break
        if err == 0:
            alpha = 1.0
        else:
            alpha = 0.5 * np.log((1.0 - err) / max(err, 1e-10))
            
        estimators.append(clf)
        estimator_weights.append(alpha)
        
        w = w * np.exp(-alpha * y_train * pred)
        w /= np.sum(w)
        
    if not estimators:
        clf = Wsvm(C=c_val, distribution_weight=np.ones(N) / N)
        clf.fit(X_train, y_train)
        return clf.predict(X_test), "None"
        
    preds = np.zeros(X_test.shape[0])
    for alpha, clf in zip(estimator_weights, estimators):
        preds += alpha * clf.predict(X_test)
        
    return np.sign(preds), "None"


def run_imada_12_decisiontree(M, X_train, y_train, X_test, theta=None, **kwargs):
    m_val = M if M is not None else 10
    theta_val = theta if theta is not None else 1.0

    clfs, alpha = ImAda_DecisionTree.fit(
        X_train,
        y_train,
        M=m_val,
        proposed_preprocessing=True,
        proposed_alpha=True,
        theta=theta_val,
        use_entropy_init=False,
        use_noise_robust_confident=False,
    )
    y_pred = ImAda_DecisionTree.predict(X_test, alpha, clfs)
    return y_pred, alpha


def run_imada_12_svm(M, C, X_train, y_train, X_test, theta=None, **kwargs):
    m_val = M if M is not None else 10
    c_val = C if C is not None else 1.0
    theta_val = theta if theta is not None else 1.0

    w, b, a = toa.fit(
        X_train,
        y_train,
        M=m_val,
        C=c_val,
        instance_categorization=False,
        proposed_preprocessing=True,
        proposed_alpha=True,
        test_something=False,
        theta=theta_val,
        use_entropy_init=False,
        use_noise_robust_confident=False,
        cleanup_gpu=True,
    )
    y_pred = toa.predict(X_test, w, b, a, M=m_val, cleanup_gpu=True)
    return y_pred, a


def run_imada_12_wsvm(M, C, X_train, y_train, X_test, theta=None, **kwargs):
    m_val = M if M is not None else 10
    c_val = C if C is not None else 1.0
    theta_val = theta if theta is not None else 1.0

    w, b, a = toa.fit(
        X_train,
        y_train,
        M=m_val,
        C=c_val,
        instance_categorization=True,
        proposed_preprocessing=True,
        proposed_alpha=True,
        test_something=False,
        theta=theta_val,
        use_entropy_init=False,
        use_noise_robust_confident=False,
        cleanup_gpu=True,
    )
    y_pred = toa.predict(X_test, w, b, a, M=m_val, cleanup_gpu=True)
    return y_pred, a

def run_fearn_adaboost_wsvm(M, C, X_train, y_train, X_test, K=9, lambda_se=1.28, mu_se=0.4, **kwargs):
    w, b, a = fearn_toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=True,
        proposed_preprocessing=True,
        use_entropy_init=False,
        use_fuzzy_spatial_weight=True,
        knn_k=K,
        lambda_se=lambda_se,
        mu_se=mu_se,
        cleanup_gpu=True,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


def run_fearn_adaboost_svm(M, C, X_train, y_train, X_test, K=9, lambda_se=1.28, mu_se=0.4, **kwargs):
    w, b, a = fearn_toa.fit(
        X_train,
        y_train,
        M,
        C,
        instance_categorization=False,
        proposed_preprocessing=True,
        use_entropy_init=False,
        use_fuzzy_spatial_weight=True,
        knn_k=K,
        lambda_se=lambda_se,
        mu_se=mu_se,
        cleanup_gpu=True,
    )
    y_pred = fearn_toa.predict(X_test, w, b, a, M)
    return y_pred, a


