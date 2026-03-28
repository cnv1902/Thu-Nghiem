import numpy as np
import methods
import svm
from sklearn import tree 

def fit(
    X,
    y,
    M=10,
    proposed_preprocessing=False,
    proposed_alpha=False,
    theta=1,
    use_entropy_init=False,
    use_noise_robust_confident=False,
):
    '''
    Input:
        X: data
        y: label
        M: Adaboost loops
        instance_categorization is  boolean which means use or not use  instance categorization
    Output H is a models of adaboosts , which is sign func of sum of M loops SVM
    '''
    #Xac dinh number of data va length of feature
    N, d = X.shape
    # initial weight adjustment and instance categorization
    if use_entropy_init:
        W_ada = methods.entropy_init_weight(X, y, proposed=proposed_preprocessing)
    else:
        W_ada = methods.intinitialization_weight_adjustment(X, y, proposed_preprocessing, theta)
    
    # W_ada = methods.intinitialization_weight_adjustment(N)
    #Creat list of each models decisiontree after adaboost
    clfs = []
    #creat list of cofident
    alpha = []
  

    for i in range(M):
        #train weak classifier with sample weight
        # weak_clf = DecisionTree(criterion='gini', max_depth=5)
        weak_clf = tree.DecisionTreeClassifier()
        weak_clf.fit(X, y, sample_weight=W_ada)

        pred_i = weak_clf.predict(X)

        true_index, false_index,false_index_P,false_index_N = methods.find_true_false_index(y, pred_i)
        # Compute i-th confident and append to the alpha
        # alpha_i = methods.confident(W_ada,false_index_P,false_index_N,proposed_alpha) #Gốc
        if use_noise_robust_confident:
            alpha_i, D_i = methods.noise_robust_confident(
                X,
                y,
                W_ada,
                false_index_P,
                false_index_N,
                proposed_alpha=proposed_alpha,
            )
        else:
            alpha_i, D_i = methods.confident(W_ada,false_index_P,false_index_N,proposed_alpha)
        alpha.append(alpha_i)
        
        clfs.append(weak_clf)
        # Update weight adjustment and instance categorization
        W_ada = methods.update_weight_adjustment(W_ada, alpha_i,true_index, false_index)
            
    return clfs, alpha   
            

# def predict(X, alpha,M =10 ):
#     H = np.zeros(X.shape[0])
#     for i in range (M):
#         H += alpha[i]*(X.dot(w[i]) +b[i])
#     return np.sign(H)

def predict(X, alpha, clfs):
    loops = min(len(alpha), len(clfs))
    if loops <= 0:
        return np.where(np.zeros(len(X)) > 0, 1, -1)

    alpha_vec = np.asarray(alpha[:loops])
    weak_preds = np.asarray([clfs[i].predict(X) for i in range(loops)])
    weak_preds = np.where(weak_preds == 1, 1, -1)

    y_score = np.dot(alpha_vec, weak_preds)
    return np.where(y_score > 0, 1, -1)