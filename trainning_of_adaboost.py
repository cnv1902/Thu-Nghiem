import numpy as np
import methods
import svm
def fit(X, y, M = 10, C = None , instance_categorization = False, proposed_preprocessing = False,proposed_alpha = False, test_something = True, theta = 1):
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
    #Creat list of each models svm after adaboost
    w = []
    b = []
    #creat list of cofident
    alpha = []
    
    if instance_categorization is True:
        B_ada = methods.intinitialization_instance_categorization(N) 
        for i in range(M):
            # Creat model
            WC = W_ada * B_ada
            if test_something == False:
                wi, bi = svm.fit(X, y, C , distribution_weight= WC)
            else: 
                wi, bi = svm.fit(X, y, C , distribution_weight= np.ones(N))
            # Append wi and bi to the list
            w.append(wi)
            b.append(bi)
            #predict the model
            pred_i = methods.predict_svm(X, wi, bi)
            # Find true, false index after training svm
            true_index, false_index,false_index_P,false_index_N = methods.find_true_false_index(y, pred_i)
            # Compute i-th confident and append to the alpha
            # alpha_i = methods.confident(W_ada,false_index_P,false_index_N,proposed_alpha) #Gốc
            alpha_i, D_i = methods.confident(W_ada,false_index_P,false_index_N,proposed_alpha)
            alpha.append(alpha_i)
            
            # Update weight adjustment and instance categorization
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i,true_index, false_index)
            B_ada = methods.update_instance_categorization_final(X, y, wi, bi)
            
    else:
        for i in range(M):
            # Creat model
            if test_something == False:
                wi, bi = svm.fit(X, y, C , distribution_weight= W_ada)
            else :
                wi, bi = svm.fit(X, y, C , distribution_weight= np.ones(N))            # Append wi and bi to the list 
            w.append(wi)
            b.append(bi)
            
            # Predict the model 
            pred_i = methods.predict_svm(X, wi, bi)
            # Find true, false index after training svm
            true_index, false_index,false_index_P,false_index_N = methods.find_true_false_index(y, pred_i)
            # Compute i-th confident and append to the alpha
            # alpha_i = methods.confident(W_ada,false_index_P,false_index_N,proposed_alpha) #Gốc
            alpha_i, D_i = methods.confident(W_ada,false_index_P,false_index_N,proposed_alpha)
            alpha.append(alpha_i)
            
            # Update weight adjustment
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i,true_index,false_index)
            
    return w, b, alpha
            

def predict(X,  w, b, alpha, M =10):
    H = np.zeros(X.shape[0])
    for i in range (M):
        H += alpha[i]*(X.dot(w[i]) +b[i])
    return np.sign(H)