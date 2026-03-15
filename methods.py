import numpy as np
# =============================================================================
# def intinitialization_weight_adjustment(N):
#     '''
#     N la so diem du lieu cua X
#     '''
#     return 
# =============================================================================

def intinitialization_weight_adjustment(X, y, proposed, theta):
    # calculate N, N_min, N_maj
    
    N, d = X.shape     
    if proposed == True:
        N_pos = np.where(y == 1)[0].shape[0]
        N_neg = np.where(y == -1)[0].shape[0]
        # calculate ep silon
        eps = N_pos/N_neg 
        delta_neg = (1-eps)**theta/N
        delta_pos = (1 - eps)**theta/(eps*N)
        # find label
        X_pos_index = np.where(y == 1)[0]
        X_neg_index = np.where(y == -1)[0]
        #calculte weight
        w = np.ones(N)/ N
        w[X_pos_index] = 1/N + delta_pos
        w[X_neg_index] = 1/N - delta_neg
    else:
        w = np.ones(N)/N
    return w #shape(N, )
    
# =============================================================================
# def intinitialization_weight_adjustment(X, y, proposed):
#     # calculate N, N_min, N_maj
#     N, d = X.shape     
#     if proposed == True:
#         N_min = np.where(y == 1)[0].shape[0]
#         N_maj = np.where(y == -1)[0].shape[0]
#         # calculate ep silon
#         eps = 1 - (N_min/ N_maj)
#         delta_min = (1 - eps)**2/N
#         delta_max = (1 - eps)/N
#         # find label
#         X_pos_index = np.where(y == 1)[0]
#         X_neg_index = np.where(y == -1)[0]
#         #calculte weight
#         w = np.ones(N)/ N
#         w[X_pos_index] = 1/N + delta_min
#         w[X_neg_index] = 1/N  -delta_max
#     else:
#         w = np.ones(N)/N
#     return w #shape(N, )
# =============================================================================
def intinitialization_instance_categorization(N):
    '''
    Input: N la so diem du lieu cua X
    Output: Vecto ban dau cua C trong bai bao 2016
    '''
    return np.ones(N)

def predict_svm(X, w, b):
    '''
    Input: tap data du lieu dau vao, X shaped (N, d)
        w, b la bo model phan lop, w shaped (d, ), b shaped ()
    Output: la gia tri predict cua lan SVM thu i
    '''
    return np.sign(X.dot(w)+b)

def find_true_false_index(y, pred):
    '''
    Tim gia tri dung sai cua moi lan phan loai
    Input: y la gia tri label cua data
        pred la gia tri sau khi phan lop
    Outpit index cua phan tu dung va sai
    '''
    true_index = np.where(y == pred)[0]
    false_index = np.where(y!= pred)[0]

    false_index_P = np.where((y!= pred)&(y==1))[0]
    false_index_N = np.where((y!= pred)&(y==-1))[0]
    return true_index, false_index, false_index_P, false_index_N


def confident(W,  false_index_P,false_index_N, proposed_alpha):
    '''
    Input: 
        W: weight adjusntment, shaped (N, 1)
        false_index: wrong predict, length <= N
    Output:
        confident of model shaped ()
    '''
    # if proposed_alpha is True: #Gốc
    #     esp_P=np.sum(W[false_index_P])
    #     esp_N=np.sum(W[false_index_N])
    #     if  (esp_N+esp_P)>0:
    #         eps=esp_N+esp_P+esp_P*(1-(esp_N+esp_P)) #Difference with paper
    #         print(eps)
    #         alpha=1/2 *np.log((1- eps)/eps)
    #         return  alpha
    #     else:
    #         return 1
    # else:
    #     if np.sum(W)>0:
    #         eps = (np.sum(W[false_index_P])+ np.sum(W[false_index_N]))/np.sum(W)
    #         print(eps)
    #         alpha=1/2 *np.log((1- eps)/eps)
    #         return  alpha
    #     else:
    #         return 1


    if proposed_alpha is True:
        esp_P=np.sum(W[false_index_P])
        esp_N=np.sum(W[false_index_N])
        if  (esp_N+esp_P)>0:
            eps=esp_N+esp_P+esp_P*(1-(esp_N+esp_P)) #Gốc
            # eps = esp_N + esp_P*(2-esp_N-esp_P)
            alpha=1/2 *np.log((1- eps)/eps)
            return  alpha,  eps
        else:
            return 1,1
    else:
        
        # if np.sum(W)>0: #Gốc
        #     eps = (np.sum(W[false_index_P])+ np.sum(W[false_index_N]))/np.sum(W)
        #     alpha=1/2 *np.log((1- eps)/eps)
        #     print("eps",eps)
        #     print("alpha",alpha)
        #     return  alpha, eps
        # else:
        #     return 1,1
        eps = (np.sum(W[false_index_P])+ np.sum(W[false_index_N]))/np.sum(W)
        if eps > 0 and eps < 1:
            eps = (np.sum(W[false_index_P])+ np.sum(W[false_index_N]))/np.sum(W)
            alpha=1/2 *np.log((1- eps)/eps)
            return  alpha, eps
        else:
            return 1,1



    

def update_weight_adjustment(W, alpha, true_index, false_index):
    '''
    Input:
        W: i-th weight adjustment
        alpha: ith_confident of Adaboost
        true_index, false_index: 
    Output:
        W (i+1)-th weight adjustment 
    '''
    W[true_index] = W[true_index]* np.exp(-1 * alpha)
    W[false_index] = W[false_index]* np.exp(alpha)
    return W/ np.sum(W)

def update_weights(weights, y_pred, y, alpha):
    # update sample weights
    y_dot_hx = np.where(y == y_pred, 1, -1) # 1 if y = h(x), -1 if y != h(x)
    new_weights = weights * np.exp(-alpha * y_dot_hx)
    new_weights /= new_weights.sum()  # normalization
    weights = new_weights
    return weights

def update_instance_categorization_final(X, y, w, b):
    # Obtain categorization_weight
    C = np.ones(X.shape[0])
    B = (X.dot(w) + b)
    A = 1 - y * (X.dot(w) + b)
    # BSV_weight
    num_of_BSV = np.where((-1 < B) & (B < 1))[0].shape[0]
    # print(num_of_BSV)
    pos_BSV = np.where((0 < B) & (B < 1))[0]
    num_of_pos_BSV = pos_BSV.shape[0]
    # print(num_of_pos_BSV)
    neg_BSV = np.where((-1 < B) & (B < 0))[0]
    num_of_neg_BSV = neg_BSV.shape[0]
    # print(num_of_neg_BSV)
    nhan_duong_BSV = np.where((-1 < B) & (B < 1) & (y == 1))[0]
    nhan_am_BSV = np.where((-1 < B) & (B < 1) & (y == -1))[0]
    if (num_of_pos_BSV != 0):
        C[nhan_duong_BSV] = num_of_BSV / (2 * (num_of_pos_BSV))
    if (num_of_neg_BSV != 0):
        C[nhan_am_BSV] = num_of_BSV / (2 * (num_of_neg_BSV))
    # SV weight
    num_of_SV = np.where((B == -1) | (B == 1))[0].shape[0]
    if (num_of_SV != 0):
        pos_SV = np.where((B == 1))[0]
        num_of_pos_SV = pos_SV.shape[0]
        nhan_duong_SV = np.where(((B == -1) | (B == 1)) & (y == 1))[0]
        nhan_am_SV = np.where(((B == -1) | (B == 1)) & (y == -1))[0]
        if (num_of_pos_SV != 0):
            C[nhan_duong_SV] = num_of_SV / (2 * num_of_pos_SV)
        neg_SV = np.where((B == -1))[0]
        num_of_neg_SV = neg_SV.shape[0]
        if (num_of_neg_SV != 0):
            C[nhan_am_SV] = num_of_SV / (2 * num_of_neg_SV)
    # positive noise
    # positive_noise = np.where(((A <= 2) & (y == 1)))[0]
    # positive_noise = np.where(((A > 2) & (y == -1)))[0]
    positive_noise = np.where(((B > 1) & (y == -1)))[0]
    num_of_positive_noise = positive_noise.shape[0]
    # num_of_positive = np.where(y == 1)[0].shape[0]
    num_of_positive = np.where(B > 0)[0].shape[0]
    if (num_of_positive != 0):
        C[positive_noise] = np.exp(num_of_positive_noise / num_of_positive)

    return C


def update_instance_categorization(X, y, w, b):
    # Obtain categorization_weight
    C = np.ones(X.shape[0])
    A = 1 - y * (X.dot(w)+b)
    # BSV_weight
    num_of_BSV = np.where((A> 0)&(A<2))[0].shape[0]
    pos_BSV = np.where((A> 0)&(A<2)&(y == 1))[0]
    num_of_pos_BSV = pos_BSV.shape[0]
    neg_BSV = np.where((A> 0)&(A<2)&(y == -1))[0]
    num_of_neg_BSV = neg_BSV.shape[0]
    if (num_of_pos_BSV != 0):
        C[pos_BSV] = num_of_BSV / (2 *(num_of_pos_BSV))
    if (num_of_neg_BSV != 0):
        C[neg_BSV] = num_of_BSV / (2 *(num_of_neg_BSV))
    #SV weight
    num_of_SV = np.where(A == 0)[0].shape[0]
    if (num_of_SV != 0):
        pos_SV = np.where((A== 0)&(y == 1))[0]
        num_of_pos_SV = pos_SV.shape[0]
        if (num_of_pos_SV != 0):
            C[pos_SV] = num_of_SV / (2 * num_of_pos_SV)
        neg_SV = np.where((A== 0)&(y == -1))[0]
        num_of_neg_SV = neg_SV.shape[0]
        if (num_of_neg_SV != 0):
            C[neg_SV] = num_of_SV / (2 * num_of_neg_SV)
    #positive noise 
    positive_noise= np.where(((A <= 2)&(y == 1)))[0]
    num_of_positive_noise = positive_noise.shape[0]
    num_of_positive = np.where(y == 1)[0].shape[0]
    C[positive_noise] = np.exp(num_of_positive_noise/num_of_positive)

    return C

