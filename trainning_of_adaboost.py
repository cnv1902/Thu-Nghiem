import numpy as np
import methods
import svm
from gpu_backend import cleanup_gpu_memory


def fit(
    X,
    y,
    M=10,
    C=None,
    instance_categorization=False,
    proposed_preprocessing=False,
    proposed_alpha=False,
    test_something=True,
    theta=1,
    use_entropy_init=False,
    use_noise_robust_confident=False,
    use_gpu=False,
    cleanup_gpu=False,
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
    #Creat list of each models svm after adaboost
    w = []
    b = []
    #creat list of cofident
    alpha = []
    unit_weight = np.ones(N)
    
    if instance_categorization is True:
        B_ada = methods.intinitialization_instance_categorization(N) 
        for i in range(M):
            # Creat model
            WC = W_ada * B_ada
            if test_something == False:
                wi, bi = svm.fit(X, y, C , distribution_weight= WC)
            else: 
                wi, bi = svm.fit(X, y, C , distribution_weight= unit_weight)
            # Append wi and bi to the list
            w.append(wi)
            b.append(bi)
            #predict the model
            pred_i = methods.predict_svm(X, wi, bi)
            # Find true, false index after training svm
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
            
            # Update weight adjustment and instance categorization
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i,true_index, false_index)
            B_ada = methods.update_instance_categorization_final(X, y, wi, bi)

            if cleanup_gpu:
                cleanup_gpu_memory()
            
    else:
        for i in range(M):
            # Creat model
            if test_something == False:
                wi, bi = svm.fit(X, y, C , distribution_weight= W_ada)
            else :
                wi, bi = svm.fit(X, y, C , distribution_weight= unit_weight)            # Append wi and bi to the list 
            w.append(wi)
            b.append(bi)
            
            # Predict the model 
            pred_i = methods.predict_svm(X, wi, bi)
            # Find true, false index after training svm
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
            
            # Update weight adjustment
            W_ada = methods.update_weight_adjustment(W_ada, alpha_i,true_index,false_index)

            if cleanup_gpu:
                cleanup_gpu_memory()
            
    return w, b, alpha
            

def predict(X,  w, b, alpha, M =10):
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
