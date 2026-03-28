import numpy as np


def _clip_eps(eps, lower=1e-12, upper=1 - 1e-12):
    return np.clip(eps, lower, upper)


def compute_fuzzy_spatial_weight(X, y, delta=1e-6):
    """
    Fuzzy spatial weight theo cong thuc:
    f_i = d(x_i, cen_opp) / (d(x_i, cen_own) + d(cen_pos, cen_neg) + delta)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]

    f = np.ones(X.shape[0], dtype=float)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return f

    cen_pos = X[pos_idx].mean(axis=0)
    cen_neg = X[neg_idx].mean(axis=0)
    d_centers = np.linalg.norm(cen_pos - cen_neg)

    own_center = np.where(y[:, None] == 1, cen_pos, cen_neg)
    opp_center = np.where(y[:, None] == 1, cen_neg, cen_pos)

    d_own = np.linalg.norm(X - own_center, axis=1)
    d_opp = np.linalg.norm(X - opp_center, axis=1)

    denom = d_own + d_centers + delta
    return d_opp / denom


def soft_margin_violation_from_margin(margin):
    """
    nu_i = 1 / (1 + exp(B_i)), voi B_i la functional margin.
    """
    margin = np.asarray(margin, dtype=float)
    margin = np.clip(margin, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(margin))


def fearn_confident(W, y, margin, fuzzy_weight=None, use_fuzzy_spatial_weight=True):
    """
    FEARN confidence:
        eps_neg = sum_{i in Negative}(w_i * nu_i)
        eps_pos = sum_{i in Positive}(w_i * nu_i * f_i)
        gamma = 2 - (eps_neg + eps_pos)
        eps_star = eps_neg + gamma * eps_pos
        alpha_star = 0.5 * ln((1 - eps_star) / eps_star)
    """
    W = np.asarray(W, dtype=float)
    y = np.asarray(y)

    if fuzzy_weight is None:
        fuzzy_weight = np.ones_like(W)
    else:
        fuzzy_weight = np.asarray(fuzzy_weight, dtype=float)

    nu = soft_margin_violation_from_margin(margin)

    neg_mask = (y == -1)
    pos_mask = (y == 1)

    eps_neg = np.sum(W[neg_mask] * nu[neg_mask])
    if use_fuzzy_spatial_weight:
        eps_pos = np.sum(W[pos_mask] * nu[pos_mask] * fuzzy_weight[pos_mask])
    else:
        eps_pos = np.sum(W[pos_mask] * nu[pos_mask])

    gamma = 2.0 - (eps_neg + eps_pos)
    eps_star = eps_neg + gamma * eps_pos

    eps_star = _clip_eps(eps_star)
    alpha_star = 0.5 * np.log((1.0 - eps_star) / eps_star)
    return alpha_star, eps_star, eps_neg, eps_pos, gamma

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

        esp_P_safe = 0.0
        esp_P_noisy = 0.0

        # Phân loại safe/noisy theo KNN bằng thao tác vectorized trên toàn bộ false_index_P.
        if len(false_index_P) > 0:
            nn = NearestNeighbors(n_neighbors=K + 1)
            nn.fit(X)
            _, indices = nn.kneighbors(X[false_index_P])

            neighbor_labels = y[indices[:, 1:]]
            negative_count = np.sum(neighbor_labels == -1, axis=1)
            noisy_mask = negative_count > (K / 2)

            false_index_P_arr = np.asarray(false_index_P)
            safe_P_indices = false_index_P_arr[~noisy_mask]
            noisy_P_indices = false_index_P_arr[noisy_mask]

            if safe_P_indices.size > 0:
                esp_P_safe = np.sum(W[safe_P_indices])
            if noisy_P_indices.size > 0:
                esp_P_noisy = np.sum(W[noisy_P_indices])

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
    B = X.dot(w) + b

    bsv_mask = (-1 < B) & (B < 1)
    pos_bsv_mask = (0 < B) & (B < 1)
    neg_bsv_mask = (-1 < B) & (B < 0)
    bsv_pos_label_mask = bsv_mask & (y == 1)
    bsv_neg_label_mask = bsv_mask & (y == -1)

    num_of_BSV = np.count_nonzero(bsv_mask)
    num_of_pos_BSV = np.count_nonzero(pos_bsv_mask)
    num_of_neg_BSV = np.count_nonzero(neg_bsv_mask)

    if num_of_pos_BSV != 0:
        C[bsv_pos_label_mask] = num_of_BSV / (2 * num_of_pos_BSV)
    if num_of_neg_BSV != 0:
        C[bsv_neg_label_mask] = num_of_BSV / (2 * num_of_neg_BSV)

    sv_mask = (B == -1) | (B == 1)
    num_of_SV = np.count_nonzero(sv_mask)
    if num_of_SV != 0:
        pos_sv_mask = (B == 1)
        neg_sv_mask = (B == -1)
        sv_pos_label_mask = sv_mask & (y == 1)
        sv_neg_label_mask = sv_mask & (y == -1)

        num_of_pos_SV = np.count_nonzero(pos_sv_mask)
        num_of_neg_SV = np.count_nonzero(neg_sv_mask)

        if num_of_pos_SV != 0:
            C[sv_pos_label_mask] = num_of_SV / (2 * num_of_pos_SV)
        if num_of_neg_SV != 0:
            C[sv_neg_label_mask] = num_of_SV / (2 * num_of_neg_SV)
    # positive noise
    # positive_noise = np.where(((A <= 2) & (y == 1)))[0]
    # positive_noise = np.where(((A > 2) & (y == -1)))[0]
    positive_noise_mask = (B > 1) & (y == -1)
    num_of_positive_noise = np.count_nonzero(positive_noise_mask)
    # num_of_positive = np.where(y == 1)[0].shape[0]
    num_of_positive = np.count_nonzero(B > 0)
    if num_of_positive != 0:
        C[positive_noise_mask] = np.exp(num_of_positive_noise / num_of_positive)

    return C


def update_instance_categorization(X, y, w, b):
    # Obtain categorization_weight
    C = np.ones(X.shape[0])
    A = 1 - y * (X.dot(w) + b)

    bsv_mask = (A > 0) & (A < 2)
    pos_bsv_mask = bsv_mask & (y == 1)
    neg_bsv_mask = bsv_mask & (y == -1)

    num_of_BSV = np.count_nonzero(bsv_mask)
    num_of_pos_BSV = np.count_nonzero(pos_bsv_mask)
    num_of_neg_BSV = np.count_nonzero(neg_bsv_mask)

    if num_of_pos_BSV != 0:
        C[pos_bsv_mask] = num_of_BSV / (2 * num_of_pos_BSV)
    if num_of_neg_BSV != 0:
        C[neg_bsv_mask] = num_of_BSV / (2 * num_of_neg_BSV)

    sv_mask = (A == 0)
    num_of_SV = np.count_nonzero(sv_mask)
    if num_of_SV != 0:
        pos_sv_mask = sv_mask & (y == 1)
        neg_sv_mask = sv_mask & (y == -1)
        num_of_pos_SV = np.count_nonzero(pos_sv_mask)
        num_of_neg_SV = np.count_nonzero(neg_sv_mask)
        if num_of_pos_SV != 0:
            C[pos_sv_mask] = num_of_SV / (2 * num_of_pos_SV)
        if num_of_neg_SV != 0:
            C[neg_sv_mask] = num_of_SV / (2 * num_of_neg_SV)

    positive_noise_mask = (A <= 2) & (y == 1)
    num_of_positive_noise = np.count_nonzero(positive_noise_mask)
    num_of_positive = np.count_nonzero(y == 1)
    C[positive_noise_mask] = np.exp(num_of_positive_noise / num_of_positive)

    return C

