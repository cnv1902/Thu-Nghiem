# ĐỀ XUẤT CẢI TIẾN THUẬT TOÁN Im.AdaBoost

## Tổng quan bài toán gốc

Luận án gốc (Võ Đức Quang, 2024) đề xuất Im.AdaBoost với 3 cải tiến so với AdaBoost chuẩn:
- **Nova 1 (ImAda-1)**: Khởi tạo trọng số thiên vị cho lớp thiểu số (`proposed_preprocessing`)
- **Nova 2 (ImAda-2)**: Công thức alpha mới tăng penalty lỗi trên lớp positive (`proposed_alpha`)
- **Instance Categorization**: Phân loại mẫu thành BSV/SV/Noise và gán trọng số theo vùng

Weak learner sử dụng: SVM tự viết (QP solver qua cvxopt), sklearn SVC, Weighted SVM (WSVM), DecisionTree.

**Phạm vi đề xuất:** Chỉ sử dụng AdaBoost, Im.AdaBoost và các thuật toán/kỹ thuật nổi tiếng, phổ biến trong machine learning (SMOTE, Cost-Sensitive Learning, Ensemble methods từ sklearn/imblearn). Không sử dụng fuzzy logic hay các thuật toán mang tính trừu tượng cao.

---

## HƯỚNG 1: CẢI TIẾN MỨC THUẬT TOÁN

### 1.1. Cải tiến công thức khởi tạo trọng số bằng Cost-Sensitive Weighting — mở rộng Nova 1

**Hạn chế hiện tại của Nova 1:**
- Công thức hiện tại: `eps = N_pos/N_neg`, `delta_pos = (1-eps)^theta / (eps*N)`, `delta_neg = (1-eps)^theta / N`
- Chỉ dựa vào tỉ lệ lớp (class ratio), không xét đến **độ khó phân loại** (difficulty) của từng mẫu
- Khi tỉ lệ mất cân bằng cực đoan (IR > 1:20), delta có thể quá lớn gây mất ổn định

**Đề xuất cải tiến — KNN-Based Difficulty Weight Initialization:**
- Ý tưởng: Dùng K-Nearest Neighbors (KNN) để đánh giá **độ khó** của mỗi mẫu trước khi vào boosting
- Với mỗi mẫu x_i, tính tỉ lệ k hàng xóm thuộc lớp đối diện: `difficulty_i = (số hàng xóm khác lớp) / k`
- Mẫu thiểu số ở vùng biên (difficulty cao, nhiều hàng xóm đa số) → tăng trọng số khởi tạo
- Mẫu thiểu số ở vùng an toàn (difficulty thấp) → giữ trọng số bình thường
- Mẫu thiểu số bị bao quanh toàn bộ bởi đa số (difficulty ≈ 1, có thể là noise) → giảm trọng số
- Công thức mới:
  ```
  w_i = 1/N + delta_class * g(difficulty_i)
  g(d) = d        nếu d < threshold_noise  (vùng biên, quan trọng)
  g(d) = d * 0.5  nếu d >= threshold_noise (noise, giảm ảnh hưởng)
  ```

**Ưu điểm:**
- KNN là thuật toán kinh điển, dễ hiểu, dễ giải thích trong bài báo
- Xác định mẫu "khó" dựa trên phân bố cục bộ — approach được dùng phổ biến trong Borderline-SMOTE (Nguyen et al., 2011)
- Giảm ảnh hưởng của noise ngay từ bước khởi tạo
- Chỉ cần `sklearn.neighbors.KNeighborsClassifier`, không cần thư viện lạ

**Tham khảo lý thuyết:** Phương pháp xác định vùng DANGER/SAFE/NOISE của Borderline-SMOTE (Han et al., 2005) — phân loại mẫu theo tỉ lệ hàng xóm khác lớp.

### 1.2. Cải tiến công thức tính alpha (Confident) — mở rộng Nova 2

**Hạn chế hiện tại của Nova 2:**
- Công thức: `eps = esp_N + esp_P * (1 - (esp_N + esp_P))`
- Tăng penalty cho lỗi trên lớp positive nhưng mức tăng là **cố định** qua mọi vòng lặp
- Không thích ứng theo tiến trình học (iteration-adaptive)

**Đề xuất cải tiến — Iteration-Adaptive Alpha:**
- Alpha tại vòng lặp t phụ thuộc vào **xu hướng lỗi qua các vòng trước**:
  - Nếu lỗi trên lớp thiểu số giảm → giảm dần penalty (thuật toán đang học tốt)
  - Nếu lỗi trên lớp thiểu số tăng/không giảm → tăng mạnh penalty
- Công thức: `eps_t = esp_N + esp_P * gamma_t` trong đó `gamma_t` là hệ số thích ứng
- Dùng **exponential moving average (EMA)** — kỹ thuật chuẩn trong time series:
  ```
  gamma_t = beta * gamma_{t-1} + (1 - beta) * (esp_P_{t-1} / (esp_P_{t-1} + esp_N_{t-1}))
  ```
  với `beta ∈ [0.5, 0.9]` là tham số EMA decay

**Ưu điểm:**
- EMA là kỹ thuật được sử dụng rộng rãi (Adam optimizer, time series, signal processing)
- Tự thích ứng thay vì gán cứng
- Tránh over-focusing vào lớp thiểu số khi thuật toán đã học tốt trên lớp đó
- Dễ implement: chỉ cần thêm 5-10 dòng code vào `methods.confident()`

### 1.3. Cải tiến Instance Categorization — Soft Margin + Sigmoid

**Hạn chế hiện tại (bug tiềm ẩn trong code gốc):**
- Phân loại BSV/SV dựa trên ngưỡng cứng: `B = X·w + b`, SV khi `B == ±1`
- **Vấn đề:** So sánh float `B == 1` trong Python gần như không bao giờ đúng → tập SV luôn rỗng
- Trọng số noise dùng `exp(num_noise / num_positive)` có thể bùng nổ exponential khi noise nhiều

**Đề xuất cải tiến — Soft Margin Categorization with Sigmoid Clipping:**
- Thay ngưỡng cứng bằng ngưỡng mềm: SV khi `|B| ∈ [1-τ, 1+τ]` với τ = 0.1 (tolerance)
- Trọng số noise: thay `exp()` bằng hàm **sigmoid clipping** — bounded trong [0, max_weight]:
  ```
  weight_noise = max_weight * sigmoid(num_noise / num_positive - shift)
  ```
  đảm bảo trọng số không bao giờ bùng nổ
- τ có thể giảm dần theo vòng lặp: `τ_t = τ_0 * decay^t`

**Ưu điểm:**
- Sigmoid là hàm kinh điển, dễ hiểu, bounded output
- Sửa bug thực tế trong code gốc (SV == ±1 không bao giờ xảy ra)
- Ngăn numerical overflow trên trọng số noise

### 1.4. Cost-Sensitive SVM thay thế WSVM — tận dụng sklearn

**Hạn chế hiện tại:**
- WSVM tự viết bằng cvxopt QP solver → chậm, khó debug, chỉ hỗ trợ linear kernel
- sklearn SVC đã có sẵn `class_weight` parameter — giải cùng bài toán cost-sensitive nhưng tối ưu hơn

**Đề xuất cải tiến — Im.AdaBoost + Cost-Sensitive SVC:**
- Thay `svm.fit()` (custom QP) bằng `sklearn.svm.SVC(kernel='linear', C=C, class_weight={-1: w_neg, 1: w_pos})`
- `w_pos` và `w_neg` tính từ W_ada (trọng số AdaBoost) và B_ada (instance categorization)
- Hoặc dùng `sample_weight` trong `.fit()` — truyền thẳng W_ada * B_ada

**Ưu điểm:**
- Nhanh hơn nhiều lần (libsvm đã tối ưu C/C++ bên dưới)
- Hỗ trợ RBF kernel, polynomial kernel — mở rộng khả năng phi tuyến
- Code gọn hơn, dễ tái sản xuất kết quả

### 1.5. Kết hợp nhiều loại Weak Learner (Heterogeneous Ensemble)

**Hạn chế hiện tại:**
- Mỗi cấu hình chỉ dùng 1 loại weak learner (SVM hoặc DecisionTree)
- Cùng loại learner → cùng dạng bias → dễ bị stuck trên cùng pattern sai

**Đề xuất cải tiến — Adaptive Weak Learner Selection:**
- Ở mỗi vòng boosting t, train cả SVM và DecisionTree trên tập weighted
- Chọn learner có weighted error (eps) thấp hơn làm weak learner cho vòng đó
- Nếu cả 2 có eps > 0.5 → dừng sớm (early stopping tự nhiên)

**Ưu điểm:**
- Tăng diversity của ensemble — nguyên tắc cốt lõi của ensemble learning (Kuncheva, 2004)
- Đã có sẵn cả 2 loại learner trong codebase (`svm.fit` và `tree.DecisionTreeClassifier`)
- Approach tương tự MultiBoost (Webb, 2000) — đã published

### 1.6. Early Stopping dựa trên Validation G-mean

**Hạn chế hiện tại:**
- Luôn chạy đủ M vòng lặp, không kiểm tra overfitting
- Khi M lớn trên dataset nhỏ → overfitting, G-mean trên test giảm

**Đề xuất cải tiến:**
- Tách 20% train thành validation set (stratified)
- Sau mỗi vòng boosting, đánh giá G-mean trên validation
- Dừng sớm nếu G-mean không cải thiện sau `patience = 3` vòng liên tiếp
- Trả về ensemble tại vòng có G-mean validation cao nhất

**Ưu điểm:**
- Kỹ thuật tiêu chuẩn trong deep learning và gradient boosting (XGBoost, LightGBM đều có)
- Giảm overfitting + giảm chi phí tính toán

---

## HƯỚNG 2: CẢI TIẾN MỨC DỮ LIỆU

### 2.1. SMOTE/Borderline-SMOTE trước khi boosting

**Hạn chế hiện tại:**
- Pipeline gốc chỉ dùng `change_rate_data` để **giảm** mẫu positive (tạo imbalance), không có kỹ thuật sinh thêm mẫu thiểu số

**Đề xuất cải tiến — Tích hợp SMOTE/Borderline-SMOTE:**
- **SMOTE** (Chawla et al., 2002): Sinh mẫu tổng hợp bằng nội suy giữa mẫu thiểu số và k hàng xóm gần nhất
- **Borderline-SMOTE** (Han et al., 2005): Chỉ sinh mẫu tại vùng biên (DANGER zone) → ít noise hơn SMOTE thường
- **ADASYN** (He et al., 2008): Adaptive — sinh thêm nhiều mẫu ở vùng khó, ít mẫu ở vùng dễ
- Áp dụng trước bước train, CHỈ trên tập train (tuyệt đối không chạm test)

**Kết hợp với Im.AdaBoost:**
- SMOTE cung cấp quân số cho lớp thiểu số → SVM tìm hyperplane tốt hơn
- Im.AdaBoost tinh chỉnh trọng số → giảm ảnh hưởng noise do SMOTE sinh ra

**Thư viện:** `imblearn.over_sampling.SMOTE`, `BorderlineSMOTE`, `ADASYN` — tất cả đã có sẵn, chỉ cần 2-3 dòng code.

### 2.2. Undersampling thông minh lớp đa số

**Đề xuất — Tomek Links / ENN Cleaning:**
- **Tomek Links** (Tomek, 1976): Xóa các cặp mẫu khác lớp gần nhau nhất → làm sạch vùng biên
- **Edited Nearest Neighbor - ENN** (Wilson, 1972): Xóa mẫu đa số mà KNN phân loại sai nó → giảm overlap
- **NearMiss** (Mani & Zhang, 2003): Chọn mẫu đa số gần nhất với thiểu số → giữ mẫu informative

**Ưu điểm:**
- Không tạo mẫu giả → tránh overfitting trên mẫu tổng hợp
- Giảm overlap giữa 2 lớp → SVM tìm hyperplane tốt hơn
- Giảm kích thước training set → QP solver của WSVM chạy nhanh hơn

**Thư viện:** `imblearn.under_sampling.TomekLinks`, `EditedNearestNeighbours`, `NearMiss`

### 2.3. Hybrid Sampling: SMOTE + Tomek Links / SMOTE + ENN

**Đề xuất — SMOTETomek / SMOTEENN:**
- Bước 1: SMOTE tạo thêm mẫu thiểu số (cân bằng lớp)
- Bước 2: Tomek Links hoặc ENN xóa các mẫu nhiễu tại biên (làm sạch)
- Bước 3: Im.AdaBoost train trên dữ liệu đã xử lý

**Đây là hướng cải tiến dữ liệu mạnh nhất** vì:
- SMOTE giải quyết thiếu quân số
- Tomek/ENN dọn dẹp noise do SMOTE vô tình sinh ra tại vùng chồng chéo
- Im.AdaBoost tiếp tục tinh chỉnh bằng cost-sensitive weighting

**Thư viện:** `imblearn.combine.SMOTETomek`, `SMOTEENN` — gọi 1 dòng code.

### 2.4. Feature Selection/Reduction trước khi boosting

**Hạn chế hiện tại:**
- Dữ liệu chỉ qua StandardScaler, không có bước chọn lọc feature
- Trong imbalanced learning, feature thừa/nhiễu có thể làm SVM tìm sai hyperplane

**Đề xuất — PCA hoặc SelectKBest trước Im.AdaBoost:**
- **PCA** (đã import sẵn trong code nhưng chưa dùng): giảm chiều, loại bỏ noise trong feature
- **SelectKBest** (sklearn): chọn k features có khả năng phân biệt lớp tốt nhất (dùng f_classif hoặc mutual_info_classif)
- **Variance Threshold**: loại feature có phương sai gần 0 (không chứa thông tin)

**Ưu điểm:**
- Giảm chiều → SVM chạy nhanh hơn, ít overfitting hơn
- Giữ lại features quan trọng nhất cho việc phân biệt lớp thiểu số

### 2.5. Normalization cải tiến — Robust Scaling thay Standard Scaling

**Hạn chế hiện tại:**
- Dùng `StandardScaler` (mean, std) → nhạy cảm với outliers
- Trong imbalanced data, outliers thường nằm ở lớp thiểu số → StandardScaler bị lệch

**Đề xuất — RobustScaler hoặc MinMaxScaler:**
- **RobustScaler** (sklearn): Dùng median và IQR thay mean/std → không bị ảnh hưởng outliers
- **MinMaxScaler** sau khi xử lý outliers bằng IQR clipping
- So sánh các phương pháp scaling để tìm phương pháp tốt nhất cho từng dataset

### 2.6. Stratified Sampling cải tiến cho Cross-Validation

**Hạn chế hiện tại:**
- Dùng `train_test_split` với `stratify=y` — chỉ 1 lần split, thiếu đánh giá variance
- Một số thí nghiệm dùng `random_state = 42` cố định → không đánh giá được độ ổn định

**Đề xuất — Repeated Stratified K-Fold:**
- Dùng `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` — chuẩn thống kê cho bài báo
- Báo cáo Mean ± Std
- Kèm theo Wilcoxon signed-rank test giữa các phương pháp

---

## TỔNG HỢP CÁC HƯỚNG CẢI TIẾN THEO ĐỘ KHẢ THI

| # | Cải tiến | Hướng | Độ khó | Tính mới | Khả năng cải thiện | Thư viện |
|---|----------|-------|--------|----------|-------------------|----------|
| 1.1 | KNN-Based Difficulty Weight Init | Thuật toán | Thấp | Cao | Cao | sklearn KNN |
| 1.2 | Iteration-Adaptive Alpha (EMA) | Thuật toán | Thấp | Cao | Trung bình | numpy |
| 1.3 | Soft Margin + Sigmoid Categorization | Thuật toán | Thấp | Trung bình | Trung bình | numpy |
| 1.4 | Cost-Sensitive SVC thay WSVM | Thuật toán | Thấp | Thấp | Trung bình | sklearn SVC |
| 1.5 | Adaptive Weak Learner Selection | Thuật toán | Trung bình | Cao | Trung bình | sklearn |
| 1.6 | Early Stopping (G-mean) | Thuật toán | Thấp | Thấp | Thấp | numpy |
| 2.1 | SMOTE / BL-SMOTE / ADASYN | Dữ liệu | Thấp | Thấp | Cao | imblearn |
| 2.2 | Tomek Links / ENN Cleaning | Dữ liệu | Thấp | Thấp | Trung bình | imblearn |
| 2.3 | SMOTETomek / SMOTEENN Hybrid | Dữ liệu | Thấp | Trung bình | Cao | imblearn |
| 2.4 | Feature Selection (PCA/SelectKBest) | Dữ liệu | Thấp | Thấp | Trung bình | sklearn |
| 2.5 | RobustScaler thay StandardScaler | Dữ liệu | Thấp | Thấp | Thấp | sklearn |
| 2.6 | Repeated Stratified K-Fold | Đánh giá | Thấp | Thấp | Thấp | sklearn |

---

## KHUYẾN NGHỊ TỔ HỢP CHO BÀI BÁO

### Tổ hợp A — "Cải tiến thuật toán Im.AdaBoost" (1 bài báo):
- **1.1** KNN-Based Difficulty Weight Init + **1.2** Adaptive Alpha + **1.3** Soft Margin Categorization
- Ý tưởng chủ đạo: *"Cải tiến toàn diện 3 thành phần cốt lõi của Im.AdaBoost bằng các kỹ thuật phổ biến"*
- Contribution: Mỗi thành phần Nova 1, Nova 2, Instance Categorization đều được cải tiến
- So sánh: AdaBoost → Im.AdaBoost gốc → Im.AdaBoost++ (đề xuất)

### Tổ hợp B — "Hybrid Data-Algorithm cho Imbalanced Classification" (1 bài báo):
- **2.3** SMOTETomek + **1.1** KNN Weight Init + Im.AdaBoost
- Ý tưởng chủ đạo: *"Kết hợp can thiệp dữ liệu (SMOTETomek) và can thiệp thuật toán (Im.AdaBoost cải tiến)"*
- Contribution: Chứng minh pipeline hybrid mạnh hơn mỗi thành phần đơn lẻ
- So sánh: SVM → SMOTE+SVM → Im.AdaBoost → SMOTETomek + Im.AdaBoost++ (đề xuất)

### Tổ hợp C — "Im.AdaBoost cải tiến toàn diện" (bài báo đầy đủ nhất):
- **2.3** SMOTETomek + **1.1** KNN Weight + **1.2** Adaptive Alpha + **1.3** Soft Margin + **1.5** Mixed Learners
- So sánh với SOTA: RUSBoost, EasyEnsemble, BalancedBagging, SMOTE+XGBoost
- Ablation study bóc tách từng thành phần
