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

def entropy_init_weight(X, y, proposed=True):
    """
    Khởi tạo trọng số Parameter-Free Im.AdaBoost sử dụng Shannon Entropy.
    Không còn cần tham số theta.
    """
    N, d = X.shape 
    w = np.ones(N) / N

    if proposed:
        # Tự động đếm và phân loại lớp thiểu số (min) / đa số (maj)
        labels, counts = np.unique(y, return_counts=True)
        
        # Nếu dữ liệu chỉ có 1 lớp (tránh lỗi logic)
        if len(labels) < 2:
            return w
            
        min_idx = np.argmin(counts)
        maj_idx = np.argmax(counts)
        
        N_min = counts[min_idx]
        N_maj = counts[maj_idx]
        min_label = labels[min_idx]
        maj_label = labels[maj_idx]
        
        # 1. Tính xác suất P_min và P_maj
        P_min = N_min / N
        P_maj = N_maj / N
        
        # 2. Tính Shannon Entropy (E)
        # Sử dụng công thức: E = -(P_min * ln(P_min) + P_maj * ln(P_maj))
        E = -(P_min * np.log(P_min) + P_maj * np.log(P_maj))
        
        # 3. Tính Hệ số Bù trừ Thích nghi (\lambda_auto)
        lambda_auto = 1 - (E / np.log(2))
        
        # 4. Tính tỷ lệ chênh lệch \delta (tương đương với 'eps' ở code cũ)
        delta_ratio = N_min / N_maj
        
        # 5. Tính lượng bù trừ \Delta cho hai lớp
        delta_maj = lambda_auto / N
        delta_min = lambda_auto / (delta_ratio * N)
        
        # 6. Cập nhật trọng số vào mảng w
        w[y == min_label] = (1 / N) + delta_min
        w[y == maj_label] = (1 / N) - delta_maj

    return w # shape(N, )
    
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

from sklearn.neighbors import NearestNeighbors

def noise_robust_confident(X, y, W, false_index_P, false_index_N, proposed_alpha=True, K=5):
    '''
    Hàm tính toán độ tin cậy kháng nhiễu (Noise-Robust Confident) bằng KNN.
    Input: 
        X, y: Dữ liệu và nhãn để chạy KNN quét vùng nhiễu.
        W: Trọng số (N, 1)
        false_index_P: index các mẫu DƯƠNG bị đoán sai
        false_index_N: index các mẫu ÂM bị đoán sai
        K: Số láng giềng để xét nhiễu (Mặc định = 5)
    '''
    if proposed_alpha is True:
        esp_N = np.sum(W[false_index_N])
        
        esp_P_safe = 0
        esp_P_noisy = 0
        
        # Tiền xử lý: Phân loại nhiễu bằng KNN cho các mẫu dương bị sai
        if len(false_index_P) > 0:
            # Dùng NearestNeighbors thay vì KNeighborsClassifier để chạy nhanh hơn
            # k=K+1 vì láng giềng gần nhất luôn là chính nó (khoảng cách = 0)
            nn = NearestNeighbors(n_neighbors=K+1)
            nn.fit(X)
            
            # Chỉ lấy các điểm dương dự đoán sai ra để đối chiếu
            distances, indices = nn.kneighbors(X[false_index_P])
            
            safe_P_indices = []
            noisy_P_indices = []
            
            for i, idx in enumerate(false_index_P):
                # Bỏ qua index đầu tiên vì đó là chính nó
                neighbor_indices = indices[i][1:]
                neighbor_labels = y[neighbor_indices]
                
                # Đếm số lượng láng giềng mang nhãn ÂM (-1)
                negative_count = np.sum(neighbor_labels == -1)
                
                # Nếu quá nửa láng giềng là Âm -> Điểm này bị nhiễu (Lọt thỏm trong vùng âm)
                if negative_count > (K / 2):
                    noisy_P_indices.append(idx)
                else:
                    safe_P_indices.append(idx)
            
            # Tính tổng trọng số lỗi cho 2 nhóm riêng biệt
            esp_P_safe = np.sum(W[safe_P_indices]) if len(safe_P_indices) > 0 else 0
            esp_P_noisy = np.sum(W[noisy_P_indices]) if len(noisy_P_indices) > 0 else 0

        # Tổng lỗi thực tế
        E_total = esp_N + esp_P_safe + esp_P_noisy
        
        if E_total > 0:
            # CÔNG THỨC CHỌN LỌC PHẠT (Selective Penalty)
            # Chỉ khuếch đại lỗi cho nhóm 'esp_P_safe'
            eps = E_total + esp_P_safe * (1 - E_total)
            
            # Bảo vệ để tránh lỗi toán học khi tính log (nếu eps >= 1)
            if eps >= 1: eps = 0.9999
                
            alpha = 1/2 * np.log((1 - eps) / eps)
            return alpha, eps
        else:
            return 1, 1
            
    else:
        # Thuật toán AdaBoost Gốc
        eps = (np.sum(W[false_index_P]) + np.sum(W[false_index_N])) / np.sum(W)
        if eps > 0 and eps < 1:
            alpha = 1/2 * np.log((1 - eps) / eps)
            return alpha, eps
        else:
            return 1, 1
    

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

