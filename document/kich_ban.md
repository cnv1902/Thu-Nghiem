# KỊCH BẢN THỬ NGHIỆM

> Tài liệu này thiết kế các thí nghiệm dựa trên **12 đề xuất cải tiến** trong `cai_tien.md`.
> Mỗi kịch bản tham chiếu rõ **mã cải tiến (CT)** và **Tổ hợp (A/B/C)** tương ứng.

---

## Thông tin chung

### Ánh xạ Cải tiến → Ký hiệu (từ cai_tien.md)

| Mã CT | Tên cải tiến | Thay thế | Hàm / Module ảnh hưởng |
|-------|-------------|----------|------------------------|
| **CT 1.1** | KNN-Based Difficulty Weight Init | Nova 1 (`proposed_preprocessing`) | `methods.proposed_preprocessing()` |
| **CT 1.2** | Iteration-Adaptive Alpha (EMA) | Nova 2 (`proposed_alpha` / `confident`) | `methods.confident()` |
| **CT 1.3** | Soft Margin + Sigmoid Instance Cat. | `update_instance_categorization()` | `methods.update_instance_categorization()` |
| **CT 1.4** | Cost-Sensitive SVC thay WSVM | `wsvm/`, `svm/` | `sklearn.svm.SVC(class_weight=...)` |
| **CT 1.5** | Adaptive Weak Learner Selection | Cố định 1 loại learner | `trainning_of_adaboost.fit()` |
| **CT 1.6** | Early Stopping (G-mean) | Luôn chạy M vòng | `trainning_of_adaboost.fit()` |
| **CT 2.1** | SMOTE / BL-SMOTE / ADASYN | Không có | Trước bước train |
| **CT 2.2** | Tomek Links / ENN Cleaning | Không có | Trước bước train |
| **CT 2.3** | SMOTETomek / SMOTEENN | Không có | Trước bước train |
| **CT 2.4** | Feature Selection (PCA/SelectKBest) | Không có | Trước bước train |
| **CT 2.5** | RobustScaler thay StandardScaler | `StandardScaler` | Preprocessing |
| **CT 2.6** | Repeated Stratified K-Fold | `train_test_split` | Evaluation strategy |

### Định nghĩa các mô hình tổ hợp (từ cai_tien.md)

| Tên mô hình | Thành phần | Mô tả |
|-------------|-----------|--------|
| **ImAda-12 gốc** | Nova 1 + Nova 2 + Instance Cat. + WSVM | Luận án gốc (Võ Đức Quang, 2024) |
| **ImAda-12++** | **CT 1.1** + **CT 1.2** + **CT 1.3** + WSVM | **Tổ hợp A** — cải tiến toàn bộ 3 thành phần thuật toán |
| **Hybrid-ImAda** | **CT 2.3** (SMOTETomek) + **CT 1.1** + ImAda gốc | **Tổ hợp B** — hybrid data + algorithm đơn giản |
| **Full-ImAda++** | **CT 2.3** + **CT 1.1** + **CT 1.2** + **CT 1.3** + **CT 1.5** | **Tổ hợp C** — mô hình đề xuất đầy đủ nhất |

### Datasets sử dụng

| Dataset | Nguồn | Số mẫu (xấp xỉ) | Số features | Đặc điểm |
|---------|-------|-------------------|-------------|-----------|
| Ecoli | UCI | 336 | 7 | Multi-class → binary, IR tự nhiên ~8.6% |
| Haberman | UCI | 306 | 3 | Ít features, overlap lớn |
| Pima Diabetes | UCI | 768 | 8 | Trung bình, IR ~35% |
| Transfusion | UCI | 748 | 4 | IR ~24% |
| Vertebral Column | UCI | 310 | 6 | IR tự nhiên ~32% |
| Co-Author | Tự thu thập | ~1000 | 7 | IR có thể điều chỉnh |
| Yeast | UCI | 1484 | 8 | Multi-class → binary |
| Page Blocks | UCI | 5473 | 10 | Lớn hơn, IR thấp |

### Imbalance Ratios kiểm tra
- IR = {1/3, 1/5, 1/7, 1/9, 1/11, 1/13, 1/15, 1/17, 1/20}
- Sử dụng `change_rate_data()` để tạo các mức IR từ dataset gốc

### Độ đo đánh giá
- **SP** (Specificity), **SE** (Sensitivity/Recall — quan trọng nhất), **G-mean** (√(SE×SP) — chính), **F1-score**, **Precision**, **Accuracy**, **AUC**, **Confusion Matrix**

### Chiến lược đánh giá (**CT 2.6**)
- **Repeated Stratified K-Fold**: K=5, Repeats=10 (`sklearn.model_selection.RepeatedStratifiedKFold`)
- **Báo cáo**: Mean ± Std
- **Kiểm định thống kê**: Wilcoxon signed-rank test (`scipy.stats.wilcoxon`)

### Tham số mặc định
| Tham số | Giá trị | Cải tiến liên quan |
|---------|---------|-------------------|
| M (vòng boosting) | 10 | — |
| C (SVM regularization) | 10000 | CT 1.4 |
| theta (weight init) | 1 | CT 1.1 |
| k (KNN neighbors) | 5 | CT 1.1 |
| beta (EMA decay) | 0.7 | CT 1.2 |
| tau (soft margin tolerance) | 0.1 | CT 1.3 |

---

## KB1: Đánh giá từng cải tiến thuật toán riêng lẻ (CT 1.1, 1.2, 1.3)

### Mục đích
Trả lời: *"Mỗi cải tiến CT 1.1 / CT 1.2 / CT 1.3 đóng góp bao nhiêu khi thay thế thành phần tương ứng của luận án gốc?"*

> **Tham chiếu cai_tien.md:** Hướng 1 — Cải tiến mức thuật toán, mục 1.1, 1.2, 1.3

### Các mô hình so sánh

| # | Ký hiệu | Weight Init | Alpha | Instance Cat. | Weak Learner | Mô tả |
|---|----------|-------------|-------|---------------|-------------|--------|
| 1 | **M0** | Đều (1/N) | Chuẩn | Không | WSVM | AdaBoost + WSVM baseline |
| 2 | **M1** | Nova 1 | Nova 2 | Gốc | WSVM | **ImAda-12 gốc** (luận án) |
| 3 | **M2** | **CT 1.1** (KNN) | Nova 2 | Gốc | WSVM | Chỉ thay weight init |
| 4 | **M3** | Nova 1 | **CT 1.2** (EMA) | Gốc | WSVM | Chỉ thay alpha |
| 5 | **M4** | Nova 1 | Nova 2 | **CT 1.3** (Soft) | WSVM | Chỉ thay instance cat. |
| 6 | **M5** | **CT 1.1** | **CT 1.2** | Gốc | WSVM | CT 1.1 + CT 1.2 |
| 7 | **M6** | **CT 1.1** | **CT 1.2** | **CT 1.3** | WSVM | = **ImAda-12++ (Tổ hợp A)** |

### Phân tích cần thực hiện
1. **So sánh cặp**: M2 vs M1 (đánh giá CT 1.1), M3 vs M1 (CT 1.2), M4 vs M1 (CT 1.3)
2. **Interaction effect**: M5 vs M2, M3 (CT 1.1 + 1.2 có tốt hơn từng cái?)
3. **Full combination**: M6 vs M1 (Tổ hợp A vs luận án gốc)

### Code mẫu
```python
# M2: Chỉ thay CT 1.1 — KNN Difficulty Weight Init
from sklearn.neighbors import KNeighborsClassifier

def knn_difficulty_weight(X, y, k=5, theta=1):
    """CT 1.1: Thay thế proposed_preprocessing() trong methods.py"""
    N = len(y)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    neighbors = knn.kneighbors(X, return_distance=False)
    difficulty = np.array([np.sum(y[neighbors[i]] != y[i]) / k for i in range(N)])
    
    N_pos, N_neg = np.sum(y == 1), np.sum(y == -1)
    eps = N_pos / N_neg
    w = np.ones(N) / N
    
    for i in np.where(y == 1)[0]:
        if difficulty[i] <= 0.7:
            delta = (1 - eps)**theta * (1 + difficulty[i]) / (eps * N)
        else:
            delta = (1 - eps)**theta * 0.5 / (eps * N)
        w[i] = 1/N + delta
    for i in np.where(y == -1)[0]:
        w[i] = 1/N - (1 - eps)**theta / N
    
    return w / w.sum()

# Trong fit(): thay W = proposed_preprocessing(X, y) bằng W = knn_difficulty_weight(X, y, k=5)
```

### Kết quả kỳ vọng
- M2 > M1 (CT 1.1 tốt hơn Nova 1, đặc biệt trên Haberman, Pima — dataset overlap lớn)
- M4 > M1 (CT 1.3 sửa bug SV rỗng → tìm đúng support vectors)
- **M6 > M2, M3, M4 > M1** (Tổ hợp A mạnh hơn bất kỳ cải tiến riêng lẻ nào)

---

## KB2: Đánh giá Tổ hợp A = ImAda-12++ (CT 1.1 + 1.2 + 1.3)

### Mục đích
Trả lời: *"ImAda-12++ (kết hợp cả 3 cải tiến thuật toán) có tạo ra synergy mạnh hơn tổng từng phần?"*

> **Tham chiếu cai_tien.md:** Khuyến nghị Tổ hợp A — "Cải tiến toàn diện 3 thành phần cốt lõi"

### Các mô hình so sánh

| # | Ký hiệu | Mô tả | Cải tiến áp dụng |
|---|----------|--------|------------------|
| 1 | **T0** | AdaBoost + SVM (sklearn) | Không |
| 2 | **T1** | AdaBoost + WSVM | Không |
| 3 | **T2** | AdaBoost + DecisionTree | Không |
| 4 | **T3** | **ImAda-12 gốc** + WSVM | Nova 1 + Nova 2 + Instance Cat. |
| 5 | **T4** | **ImAda-12++** + WSVM | **CT 1.1 + CT 1.2 + CT 1.3** |
| 6 | **T5** | **ImAda-12++** + CS-SVC (**CT 1.4**) | CT 1.1 + CT 1.2 + CT 1.3 + **CT 1.4** |
| 7 | **T6** | **ImAda-12++** + Adaptive Learner (**CT 1.5**) | CT 1.1 + CT 1.2 + CT 1.3 + **CT 1.5** |

### Code mẫu cho T5 (ImAda-12++ + Cost-Sensitive SVC — CT 1.4)
```python
from sklearn.svm import SVC

# Trong fit(), vòng boosting thứ t:
# Thay svm.fit(X, y, C, distribution_weight) bằng:
W_combined = W_ada * B_ada  # trọng số AdaBoost × Instance Categorization
clf = SVC(kernel='linear', C=C, class_weight={-1: 1.0, 1: N_neg/N_pos})
clf.fit(X, y, sample_weight=W_combined)

# predict:
y_pred = clf.predict(X_test)
```

### Code mẫu cho T6 (CT 1.5 — Adaptive Weak Learner)
```python
from sklearn.tree import DecisionTreeClassifier

# Mỗi vòng t: train cả 2 learner, chọn learner có eps thấp hơn
svm_clf = SVC(kernel='linear', C=C)
svm_clf.fit(X, y, sample_weight=W_combined)
eps_svm = np.sum(W_combined[svm_clf.predict(X) != y])

dt_clf = DecisionTreeClassifier(max_depth=3)
dt_clf.fit(X, y, sample_weight=W_combined)
eps_dt = np.sum(W_combined[dt_clf.predict(X) != y])

if eps_svm <= eps_dt:
    learner_t = svm_clf
else:
    learner_t = dt_clf
```

### Kết quả kỳ vọng
- **T4 > T3**: ImAda-12++ vượt luận án gốc — contribution chính
- T5 ≈ T4 hoặc T5 > T4: CS-SVC nhanh hơn + có thể tốt hơn WSVM
- T6 > T4: Adaptive learner tăng diversity

### Biểu đồ
- **Radar chart**: T0 vs T3 vs T4 trên 5 metrics (SP, SE, G-mean, F1, AUC)
- **Bar chart**: G-mean trên tất cả datasets

---

## KB3: Đánh giá cải tiến dữ liệu + ImAda-12++ (Tổ hợp B & C)

### Mục đích
Trả lời: *"Kết hợp can thiệp dữ liệu (CT 2.1-2.3) với ImAda-12++ có tạo pipeline mạnh hơn không?"*

> **Tham chiếu cai_tien.md:**
> - Hướng 2: CT 2.1, 2.2, 2.3
> - Khuyến nghị **Tổ hợp B**: SMOTETomek + CT 1.1 + ImAda gốc
> - Khuyến nghị **Tổ hợp C**: SMOTETomek + CT 1.1 + CT 1.2 + CT 1.3 + CT 1.5

### Các mô hình so sánh

| # | Ký hiệu | Sampling (CT 2.x) | Thuật toán | Tổ hợp | Mô tả |
|---|----------|-------------------|------------|--------|--------|
| 1 | **H0** | Không | SVM | — | Baseline |
| 2 | **H1** | Không | ImAda-12 gốc | — | Luận án gốc |
| 3 | **H2** | Không | **ImAda-12++** | **A** | Chỉ cải tiến thuật toán |
| 4 | **H3** | SMOTE (**CT 2.1**) | SVM | — | Chỉ can thiệp dữ liệu |
| 5 | **H4** | SMOTE (**CT 2.1**) | ImAda-12 gốc | — | SMOTE + luận án gốc |
| 6 | **H5** | SMOTE (**CT 2.1**) | **ImAda-12++** | — | SMOTE + cải tiến thuật toán |
| 7 | **H6** | BL-SMOTE (**CT 2.1**) | **ImAda-12++** | — | BL-SMOTE + ImAda-12++ |
| 8 | **H7** | SMOTETomek (**CT 2.3**) | ImAda-12 gốc | **≈B** | **Tổ hợp B** (simplified) |
| 9 | **H8** | SMOTETomek (**CT 2.3**) | **ImAda-12++** | **C** | **Tổ hợp C = Full-ImAda++** |
| 10 | **H9** | SMOTEENN (**CT 2.3**) | **ImAda-12++** | — | SMOTEENN + ImAda-12++ |
| 11 | **H10** | ADASYN (**CT 2.1**) | **ImAda-12++** | — | ADASYN + ImAda-12++ |

### Điểm khác biệt so với kich_ban cũ
- **H5, H6, H8, H9, H10 dùng ImAda-12++ (đã cải tiến)**, không chỉ ImAda-12 gốc
- So sánh trực tiếp: H4 (sampling + gốc) vs **H5** (sampling + cải tiến) → đo đúng đóng góp của cải tiến thuật toán khi đã có sampling
- H7 vs **H8** → đo đóng góp ImAda-12++ trên cùng sampling

### Code mẫu pipeline Tổ hợp C (H8)
```python
from imblearn.combine import SMOTETomek

# Trong mỗi fold:
X_train_fold, X_test_fold, y_train_fold, y_test_fold = ...  # stratified split

# Bước 1: CT 2.3 — SMOTETomek (chỉ trên train!)
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train_fold, y_train_fold)

# Bước 2: CT 1.1 — KNN Difficulty Weight Init (trên dữ liệu đã resample)
W = knn_difficulty_weight(X_resampled, y_resampled, k=5, theta=1)

# Bước 3: Boosting loop với CT 1.2 (EMA Alpha) + CT 1.3 (Soft Margin)
eps_history = []
for t in range(M):
    w_t, b_t = wsvm.fit(X_resampled, y_resampled, C, W * B_ada)
    
    # CT 1.3: Soft Margin Instance Categorization
    B_ada = update_instance_categorization_soft(X_resampled, y_resampled, w_t, b_t, tau=0.1)
    
    # CT 1.2: Adaptive Alpha
    alpha_t, eps_t = confident_adaptive(W, false_P, false_N, eps_history, beta=0.7)
    
    W = update_weight(W, alpha_t, ...)

# Bước 4: Evaluate trên test GỐC (không sampling!)
y_pred = predict(X_test_fold, w_list, b_list, alpha_list, M)
```

### Kết quả kỳ vọng
- **H2 > H1**: ImAda-12++ > gốc (chỉ algorithm)
- **H5 > H4 > H3**: SMOTE + ImAda-12++ > SMOTE + gốc > SMOTE + SVM
- **H8 > H7**: SMOTETomek + ImAda-12++ > SMOTETomek + gốc (Tổ hợp C > Tổ hợp B)
- **H8 > H2**: Hybrid (Tổ hợp C) > Algorithm-only (Tổ hợp A)
- **H8 tốt nhất** — đây là mô hình đề xuất cuối cùng (Full-ImAda++)

---

## KB4: Ablation Study — Bóc tách từng cải tiến của Tổ hợp C

### Mục đích
Chứng minh **mỗi CT** trong Tổ hợp C đều đóng góp, không thừa.

> **Tham chiếu cai_tien.md:** Tổ hợp C = CT 2.3 + CT 1.1 + CT 1.2 + CT 1.3 + CT 1.5

### Cấu hình Ablation (cộng dồn theo đúng thứ tự cải tiến)

| Config | CT áp dụng | Sampling | Weight Init | Alpha | Instance Cat. | Learner | Mô tả |
|--------|-----------|----------|-------------|-------|---------------|---------|--------|
| **A0** | Không | Không | Đều (1/N) | Chuẩn | Không | WSVM | AdaBoost + WSVM baseline |
| **A1** | Nova 1+2+Cat | Không | Nova 1 | Nova 2 | Gốc | WSVM | = ImAda-12 gốc (luận án) |
| **A2** | + **CT 1.1** | Không | **KNN Difficulty** | Nova 2 | Gốc | WSVM | Thay Nova 1 bằng CT 1.1 |
| **A3** | + **CT 1.2** | Không | KNN Difficulty | **EMA Alpha** | Gốc | WSVM | + Thay Nova 2 bằng CT 1.2 |
| **A4** | + **CT 1.3** | Không | KNN Difficulty | EMA Alpha | **Soft Margin** | WSVM | = **ImAda-12++ (Tổ hợp A)** |
| **A5** | + **CT 2.3** | **SMOTETomek** | KNN Difficulty | EMA Alpha | Soft Margin | WSVM | + Data-level |
| **A6** | + **CT 1.5** | SMOTETomek | KNN Difficulty | EMA Alpha | Soft Margin | **Adaptive** | = **Full-ImAda++ (Tổ hợp C)** |

### Bảng kết quả mẫu

| Config | CT mới thêm | Mô tả | G-mean | ΔG-mean | SE | F1 | Time (s) |
|--------|------------|--------|--------|---------|-----|-----|----------|
| A0 | — | AdaBoost baseline | ... | — | ... | ... | ... |
| A1 | Nova 1+2+Cat | ImAda-12 gốc | ... | +Δ1 | ... | ... | ... |
| A2 | **CT 1.1** | + KNN weight | ... | +Δ2 | ... | ... | ... |
| A3 | **CT 1.2** | + EMA alpha | ... | +Δ3 | ... | ... | ... |
| A4 | **CT 1.3** | = ImAda-12++ | ... | +Δ4 | ... | ... | ... |
| A5 | **CT 2.3** | + SMOTETomek | ... | +Δ5 | ... | ... | ... |
| A6 | **CT 1.5** | = **Full-ImAda++** | ... | +Δ6 | ... | ... | ... |

### Kết quả kỳ vọng
- **Mỗi Δ > 0** — chứng minh mỗi CT đều cần thiết
- A1 → A2 (CT 1.1): bước nhảy trên dataset overlap (Haberman, Pima)
- A3 → A4 (CT 1.3): bước nhảy do sửa bug SV rỗng
- A4 → A5 (CT 2.3): bước nhảy lớn nhất (thêm quân số cho thiểu số)
- A5 → A6 (CT 1.5): bước nhảy nhỏ nhưng ổn định (tăng diversity)

### Ablation ngược (loại bỏ từng CT khỏi Tổ hợp C)

| Config | CT bị loại | G-mean | ΔG so với A6 |
|--------|-----------|--------|-------------|
| A6 | Đầy đủ | ... | 0 (reference) |
| A6 − CT 1.1 | Bỏ KNN weight, giữ Nova 1 | ... | −Δ |
| A6 − CT 1.2 | Bỏ EMA, giữ Nova 2 | ... | −Δ |
| A6 − CT 1.3 | Bỏ Soft, giữ Cat. gốc | ... | −Δ |
| A6 − CT 2.3 | Bỏ SMOTETomek | ... | −Δ |
| A6 − CT 1.5 | Bỏ Adaptive learner | ... | −Δ |

→ CT nào bị loại mà G-mean giảm nhiều nhất = CT quan trọng nhất.

---

## KB5: Stress Test — ImAda-12++ vs Baselines trên IR cực đoan

### Mục đích
Chứng minh **Tổ hợp C (Full-ImAda++)** bền hơn khi IR ngày càng cực đoan.

> **Tham chiếu cai_tien.md:** Tổ hợp C và đánh giá ở nhiều mức IR

### Cách thức thực hiện
1. Dataset: Co-Author (điều chỉnh IR qua `change_rate_data`)
2. IR = {20%, 15%, 10%, 8%, 6%, 4%, 2%}
3. **7 mô hình so sánh, mỗi mô hình ánh xạ rõ CT:**

| # | Ký hiệu | Mô tả | CT áp dụng |
|---|----------|--------|-----------|
| 1 | AdaBoost | sklearn AdaBoostClassifier | Không |
| 2 | AdaBoost+WSVM | Codebase gốc, không cải tiến | Không |
| 3 | ImAda-12 | Luận án gốc | Nova 1 + Nova 2 + Cat. |
| 4 | SMOTE+SVM | Chỉ data-level | CT 2.1 |
| 5 | SMOTETomek+ImAda-12 | **Tổ hợp B** | CT 2.3 + Nova gốc |
| 6 | **ImAda-12++** | **Tổ hợp A** (algorithm-only) | **CT 1.1 + CT 1.2 + CT 1.3** |
| 7 | **Full-ImAda++** | **Tổ hợp C** (full) | **CT 2.3 + CT 1.1 + CT 1.2 + CT 1.3 + CT 1.5** |

4. Repeated Stratified 5-fold × 10 lần

### Biểu đồ
- **Line chart with error bars**: trục X = IR (%), trục Y = G-mean → Full-ImAda++ phẳng nhất
- **Bar chart**: SE tại IR = 4% và 2% → baselines sụt, Full-ImAda++ giữ vững
- **So sánh Tổ hợp A vs B vs C** tại mỗi IR → xác định tổ hợp tốt nhất

---

## KB6: Đánh giá Cost-Sensitive SVC thay WSVM (CT 1.4)

### Mục đích
Trả lời: *"Dùng sklearn SVC (class_weight) thay WSVM tự viết (cvxopt) có nhanh hơn và tốt hơn không?"*

> **Tham chiếu cai_tien.md:** Cải tiến 1.4 — Cost-Sensitive SVM thay thế WSVM

### Các mô hình so sánh

| # | Ký hiệu | Weak Learner | Thuật toán boosting | Mô tả |
|---|----------|-------------|-------------------|--------|
| 1 | **L1** | WSVM (cvxopt) | ImAda-12 gốc | Luận án gốc |
| 2 | **L2** | sklearn SVC (linear) | ImAda-12 gốc | Chỉ thay learner |
| 3 | **L3** | sklearn SVC (linear, **CT 1.4** class_weight) | ImAda-12 gốc | + Cost-sensitive |
| 4 | **L4** | WSVM (cvxopt) | **ImAda-12++** | Tổ hợp A + WSVM |
| 5 | **L5** | sklearn SVC (linear, **CT 1.4**) | **ImAda-12++** | Tổ hợp A + CS-SVC |
| 6 | **L6** | sklearn SVC (**rbf**, CT 1.4) | **ImAda-12++** | RBF kernel (mở rộng) |

### Metrics bổ sung
- **Training Time (s)**: Quan trọng — SVC(libsvm) nên nhanh hơn WSVM(cvxopt) đáng kể
- So sánh L4 vs L5: cùng ImAda-12++, khác learner → isolate đóng góp CT 1.4
- L6: thử RBF kernel — ưu thế mà WSVM tự viết không có

### Kết quả kỳ vọng
- L5 ≈ L4 (G-mean tương đương) nhưng L5 nhanh hơn nhiều → recommend CS-SVC
- L6 có thể tốt hơn trên dataset phi tuyến (Pima, Haberman)

---

## KB7: So sánh Full-ImAda++ (Tổ hợp C) với State-of-the-Art

### Mục đích
So sánh mô hình đề xuất cuối cùng (**Full-ImAda++ = Tổ hợp C**) với các SOTA nổi tiếng.

> **Tham chiếu cai_tien.md:** Tổ hợp C — "Im.AdaBoost cải tiến toàn diện"

### Các mô hình so sánh

| # | Ký hiệu | Phương pháp | Nguồn | CT tương đương |
|---|----------|------------|-------|---------------|
| 1 | **S1** | SVM (class_weight='balanced') | `sklearn.svm.SVC` | Tương tự CT 1.4 |
| 2 | **S2** | AdaBoost chuẩn | `sklearn.ensemble.AdaBoostClassifier` | — |
| 3 | **S3** | SMOTE + Random Forest | `imblearn` + `sklearn` | CT 2.1 |
| 4 | **S4** | SMOTE + XGBoost | `imblearn` + `xgboost` | CT 2.1 |
| 5 | **S5** | RUSBoost | `imblearn.ensemble.RUSBoostClassifier` | —  |
| 6 | **S6** | EasyEnsemble | `imblearn.ensemble.EasyEnsembleClassifier` | — |
| 7 | **S7** | BalancedBagging | `imblearn.ensemble.BalancedBaggingClassifier` | — |
| 8 | **S8** | BalancedRandomForest | `imblearn.ensemble.BalancedRandomForestClassifier` | — |
| 9 | **S9** | **ImAda-12 gốc** | Codebase | Nova 1 + 2 + Cat. |
| 10 | **S10** | **ImAda-12++ (Tổ hợp A)** | Cải tiến | CT 1.1 + 1.2 + 1.3 |
| 11 | **S11** | **Full-ImAda++ (Tổ hợp C)** | Cải tiến | CT 2.3 + 1.1 + 1.2 + 1.3 + 1.5 |

### Lưu ý
- Tất cả chạy trên **cùng folds** (cùng `random_state`) — công bằng
- S5-S8 là **SOTA ensemble cho imbalanced** — reviewer bắt buộc kỳ vọng thấy
- S10 cho thấy đóng góp thuần thuật toán, S11 là full model

### Kết quả kỳ vọng
- **S11 đạt G-mean, SE, F1 cao nhất** hoặc cạnh tranh top 1-2
- S10 > S9 → chứng minh cải tiến thuật toán hiệu quả
- S11 > S10 → chứng minh thêm CT 2.3 + CT 1.5 vẫn tạo thêm giá trị
- Friedman test + Nemenyi post-hoc xếp hạng trung bình qua tất cả datasets

### Bảng kết quả mẫu

| Method | SP | SE | G-mean | F1 | AUC | Rank |
|--------|-----|-----|--------|-----|------|------|
| S1 - Balanced SVM | ... | ... | ... | ... | ... | ... |
| S2 - AdaBoost | ... | ... | ... | ... | ... | ... |
| S3 - SMOTE+RF | ... | ... | ... | ... | ... | ... |
| S4 - SMOTE+XGB | ... | ... | ... | ... | ... | ... |
| S5 - RUSBoost | ... | ... | ... | ... | ... | ... |
| S6 - EasyEnsemble | ... | ... | ... | ... | ... | ... |
| S7 - BalancedBagging | ... | ... | ... | ... | ... | ... |
| S8 - BalancedRF | ... | ... | ... | ... | ... | ... |
| S9 - ImAda-12 gốc | ... | ... | ... | ... | ... | ... |
| **S10 - ImAda-12++ (Tổ hợp A)** | ... | ... | ... | ... | ... | ... |
| **S11 - Full-ImAda++ (Tổ hợp C)** | ... | ... | ... | ... | ... | ... |

---

## KB8: Phân tích độ nhạy siêu tham số của ImAda-12++ (Sensitivity Analysis)

### Mục đích
Khảo sát ảnh hưởng từng siêu tham số **mới** được giới thiệu bởi các CT → xác định vùng ổn định.

> **Tham chiếu cai_tien.md:** Tham số mới: k (CT 1.1), beta (CT 1.2), tau (CT 1.3)

### Tham số khảo sát (ánh xạ CT)

| Tham số | CT liên quan | Ý nghĩa | Dải giá trị | Default |
|---------|-------------|---------|-------------|---------|
| k | **CT 1.1** | Số neighbors KNN difficulty | [3, 5, 7, 9, 11] | 5 |
| beta | **CT 1.2** | EMA decay cho adaptive alpha | [0.3, 0.5, 0.7, 0.8, 0.9] | 0.7 |
| tau | **CT 1.3** | Soft margin tolerance | [0.01, 0.05, 0.1, 0.2, 0.3] | 0.1 |
| M | Gốc | Số vòng boosting | [5, 10, 15, 20, 25, 30, 40, 50] | 10 |
| C | Gốc | SVM regularization | [0.1, 1, 10, 100, 1000, 5000, 10000] | 10000 |
| theta | **CT 1.1** | Hệ số điều chỉnh weight init | [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5] | 1.0 |

### Cách thức
- Mô hình: **ImAda-12++ (Tổ hợp A)** — để đo riêng ảnh hưởng tham số thuật toán
- Fix tất cả tham số khác ở giá trị default, biến thiên 1 tham số
- 3 datasets: Ecoli (nhỏ), Pima (trung bình), Co-Author (lớn)
- 5-fold CV × 5 lần

### Biểu đồ cần vẽ
- **6 subplot**: mỗi tham số 1 đường G-mean (3 đường cho 3 datasets)
- Xác định vùng **plateau** → tham số ổn định, không nhạy
- Đánh dấu giá trị default trên mỗi đường

---

## Tổng hợp: Ánh xạ Kịch bản ↔ Cải tiến ↔ Tổ hợp

| KB | Tên | CT được đánh giá | Tổ hợp liên quan | Câu hỏi chính |
|----|-----|-----------------|------------------|---------------|
| **KB1** | Từng CT thuật toán | CT 1.1, 1.2, 1.3 riêng lẻ | Thành phần của A | Mỗi CT đóng góp bao nhiêu? |
| **KB2** | Tổ hợp A + mở rộng | CT 1.1+1.2+1.3, CT 1.4, CT 1.5 | **A** + mở rộng | ImAda-12++ + CS-SVC + Adaptive? |
| **KB3** | Data + Algorithm | CT 2.1-2.3 + ImAda-12++ | **B** và **C** | Hybrid pipeline tốt hơn? |
| **KB4** | Ablation Study | Tất cả CT cộng dồn | **C** (bóc tách) | Mỗi CT có cần thiết? |
| **KB5** | Stress Test IR | Tổ hợp A vs B vs C | **A, B, C** | Bền trên IR cực đoan? |
| **KB6** | Cost-Sensitive SVC | CT 1.4 chuyên sâu | Phần của A | SVC thay WSVM? |
| **KB7** | SOTA comparison | Tổ hợp A và C | **A** và **C** vs SOTA | Cạnh tranh top? |
| **KB8** | Sensitivity | k, beta, tau, M, C, theta | Params của A | Vùng ổn định? |

## Thứ tự ưu tiên chạy thực nghiệm

| Ưu tiên | Kịch bản | Lý do |
|---------|----------|-------|
| **1** | **KB1** (Từng CT riêng lẻ) | Validate từng cải tiến, nền tảng cho mọi KB khác |
| **2** | **KB2** (Tổ hợp A) | Xác nhận ImAda-12++ mạnh hơn gốc — contribution chính |
| **3** | **KB3** (Tổ hợp B & C) | Thêm data-level → Full-ImAda++ — mô hình cuối cùng |
| **4** | **KB4** (Ablation) | Bắt buộc cho bài báo, dùng kết quả KB1-3 |
| **5** | **KB6** (CS-SVC) | Thay thế WSVM → đơn giản hóa + tốc độ |
| **6** | **KB5** (Stress Test) | Biểu đồ ấn tượng, kiểm chứng robustness |
| **7** | **KB7** (SOTA) | So sánh công bằng với 8 phương pháp nổi tiếng |
| **8** | **KB8** (Sensitivity) | Chạy cuối khi đã chốt mô hình và tham số |

---

## Thư viện cần cài thêm

```bash
pip install imbalanced-learn   # CT 2.1-2.3: SMOTE, BL-SMOTE, SMOTETomek, SMOTEENN + KB7: RUSBoost, EasyEnsemble
pip install xgboost            # KB7: So sánh SOTA
pip install scipy              # Wilcoxon test, Friedman test
pip install matplotlib         # Biểu đồ
pip install seaborn            # Heatmap
```

---

## Cấu trúc output thí nghiệm

```
Experiment/
├── KB1_Individual_CT/
│   ├── CT1.1_KNN_weight/           # M2 vs M1
│   ├── CT1.2_EMA_alpha/            # M3 vs M1
│   └── CT1.3_Soft_margin/          # M4 vs M1
├── KB2_Combination_A/
│   ├── ImAda12pp_vs_original/      # T4 vs T3
│   ├── CS_SVC/                     # T5 vs T4
│   └── Adaptive_learner/           # T6 vs T4
├── KB3_Hybrid_Pipeline/
│   ├── Sampling_comparison/        # H3-H10
│   └── Combination_B_vs_C/         # H7 vs H8
├── KB4_Ablation/
│   ├── additive_ablation.csv       # A0 → A6
│   └── removal_ablation.csv        # A6 − CT_x
├── KB5_Stress_Test/
│   └── Co_Author_IR_sweep.csv
├── KB6_Cost_Sensitive_SVC/
│   └── WSVM_vs_SVC_comparison.csv
├── KB7_SOTA/
│   └── all_methods_all_datasets.csv
└── KB8_Sensitivity/
    ├── param_k_CT1.1.csv
    ├── param_beta_CT1.2.csv
    ├── param_tau_CT1.3.csv
    ├── param_M.csv
    ├── param_C.csv
    └── param_theta.csv
```

Mỗi file CSV có header chuẩn:
```
Dataset, IR, Method, CT_Applied, Fold, Run, SP, SE, Gmean, F1, Precision, Accuracy, AUC, ConfusionMatrix, TrainingTime
```
