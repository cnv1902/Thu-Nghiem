import numpy as np
import methods
import svm
from sklearn import tree 

def fit(X, y, M = 10, proposed_preprocessing = False, proposed_alpha = False, theta = 1):
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
    y_pred = np.zeros(len(X))
    for alpha, clf in zip(alpha, clfs):
        y_pred_weak = clf.predict(X)
        # quantize y_pred_weak to {0, 1}
        y_pred_weak = np.where(y_pred_weak == 1, 1, -1)
        y_pred += alpha * y_pred_weak

    # quantize y_pred to {-1, 1}
    y_pred = np.where(y_pred > 0, 1, -1)
    return y_pred