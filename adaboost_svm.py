# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 07:54:27 2020

@author: DELL
"""
from typing import BinaryIO
import numpy as np
import methods
from sklearn.svm import SVC
from data.Vertebral_column import load_data
from sklearn.metrics import classification_report

def fit(X, y, M = 10, C = None , instance_categorization = False, proposed = False, theta=None):
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
    #W_ada = methods.intinitialization_weight_adjustment(X, y, proposed)
    W_ada = methods.intinitialization_weight_adjustment(X, y, proposed, theta)
    # W_ada = methods.intinitialization_weight_adjustment(N)
    #Creat list of each models svm after adaboost
    w = []
    b = []
    #creat list of cofident
    alpha = []
    if instance_categorization is True:
        C_ada = methods.intinitialization_instance_categorization(N)
        for i in range(M):
            # Creat model
            WC = W_ada * C_ada
            clf = SVC.fit(kernel= 'linear', C = 10000, class_weight = WC)   
            wi = clf.coef_
            bi = clf.intercept_[0]
            # Append wi and bi to the list
            w.append(wi)
            b.append(bi)
            #predict the model
            pred_i = methods.predict_svm(X, wi, bi)
            # Find true, false index after training svm
            true_index, false_index,false_index_P,false_index_N = methods.find_true_false_index(y, pred_i)
            # Compute i-th confident and append to the alpha
            alpha_i = methods.confident(W_ada,false_index,false_index_P,false_index_N)
            alpha.append(alpha_i)
            # Update weight adjustment and instance categorization
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i,true_index, false_index)
            C_ada = methods.update_instance_categorization(X, y, wi, bi)
    else:
        for i in range(M):
            # Creat model
            
            clf = SVC(kernel= 'linear', C = 10000)
            clf.fit(X, y, W_ada)
            wi =  clf.coef_.flatten()
            bi = clf.intercept_[0]
            # Append wi and bi to the list 
            w.append(wi)
            b.append(bi)
            # Predict the model 
            pred_i = methods.predict_svm(X, wi, bi)
            # Find true, false index after training svm
            true_index, false_index, false_index_P,false_index_N = methods.find_true_false_index(y, pred_i)
            # Compute i_th confident and append to the alpha
            alpha_i = methods.confident(W_ada,false_index,false_index_P,false_index_N)
            alpha.append(alpha_i)
            # Update weight adjustment
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i,true_index,false_index)
    return w, b, alpha    

def predict(X,  w, b, alpha,M =10 ):
    H = np.zeros(X.shape[0])
    for i in range (M):
        H += alpha[i]*(X.dot(w[i]) +b[i])
    return np.sign(H)

# X_train, y_train,X_test, y_test = load_data()
# w, b, alpha = fit(X_train, y_train)
# pred_y = predict(X_test, w, b, alpha)
# print(classification_report(y_test, pred_y))


