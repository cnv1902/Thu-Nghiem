# common
## change_rate_data.py
1. change_rate_data:
- input: (X, y, new_rate)
- output: (X, y) với tỉ lệ mất cân bằng (+1/-1) nhãn theo new_rate bằng giữ lại toàn bộ nhãn âm (-1), random và giảm số lượng mẫu nhãn dương (+1)
# data
## common
### change_rate_data.py
1. change_rate_data:
- input: (X, y, new_rate)
- output: (X, y) với tỉ lệ mất cân bằng (+1/-1) nhãn theo new_rate bằng giữ lại toàn bộ nhãn âm (-1), random và giảm số lượng mẫu nhãn dương (+1) và in ra số lượng mẫu nhãn dương được dữ lại
2. change_rate_data_cnn:
- input: (X, y, new_rate)
- output: (X, y) với tỉ lệ mất cân bằng (+1/0) nhãn theo new_rate bằng giữ lại toàn bộ nhãn âm (0), random và giảm số lượng mẫu nhãn dương (+1)
## dataset
### change_rate_data.py
1. change_rate_data:
- input: (X, y, new_rate)
- output: (X, y) với tỉ lệ mất cân bằng (+1/total) nhãn theo new_rate bằng giữ lại toàn bộ nhãn âm (-1), random và giảm số lượng mẫu nhãn dương (+1)
### Abanole_CNN.py
1. load_data:
- input: (test_size)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset abalone, encode Sex bằng LabelEncoder, chuyển Rings → nhị phân (15=1, khác=0), chia train/test theo test_size (stratify) và encode nhãn bằng to_categorical.
### Abanole_TestSize.py
1. load_data:
- input: test_size
- output: (X_train, y_train, X_test, y_test) đọc dataset abalone, chuyển nhãn Rings → binary (15 = 1, còn lại = -1), encode cột Sex bằng LabelEncoder, chia train/test với stratify, chuẩn hóa dữ liệu X bằng StandardScaler
### Vertebral_column.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset Vertebral Column, chuyển nhãn Abnormal → -1, Normal → 1, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify)
### Vertebral_column_KF.py
1. load_data:
- input: ()
- output: (X, y) đọc dataset Vertebral Column, chuyển nhãn Abnormal → -1, Normal → 1, trả về toàn bộ X, y (dùng cho KFold)
### Co_Author.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset Co_Author_100_500_1, đảo nhãn (-1 → 1, 1 → -1), thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Co_Author_KF.py
1. load_data:
- input: ()
- output: (X, y) đọc dataset Co_Author_100_500_1, đảo nhãn (-1 → 1, 1 → -1), trả về toàn bộ X, y (dùng cho KFold)
### Co_Author_CNN.py
1. load_data:
- input: (test_size)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset Co_Author_100_500_1, chuyển nhãn (-1 → 0, 1 → 1), chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Co_Author_CNN_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset CoAuthor_800_4000, chuyển nhãn (-1 → 0, 1 → 1), thay đổi tỉ lệ mất cân bằng bằng change_rate_data_cnn, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Co_Author_TestSize.py
1. load_data:
- input: (test_size)
- output: (X_train, y_train, X_test, y_test) đọc dataset Co-Author-100-500-st-unw, giữ nguyên nhãn, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Co_Author_TestSize_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset CoAuthor_500_2500, giữ nguyên nhãn, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Ecoli_Kfold.py
1. load_data:
- input: ()
- output: (X, y) đọc dataset ecoli_new, chuyển nhãn im → 1, các lớp khác → -1, trả về toàn bộ X, y (dùng cho KFold)
### Ecoli_CNN.py
1. load_data:
- input: (test_size)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset ecoli_new, chuyển nhãn im → 1, các lớp khác → 0, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Ecoli_CNN_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset ecoli_new, chuyển nhãn im → 1, các lớp khác → 0, thay đổi tỉ lệ mất cân bằng bằng change_rate_data_cnn, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Ecoli_TestSize.py
1. load_data:
- input: (test_size)
- output: (X_train, y_train, X_test, y_test) đọc dataset ecoli_new, chuyển nhãn im → 1, các lớp khác → -1, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Ecoli_TestSize_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset ecoli_new, chuyển nhãn im → 1, các lớp khác → -1, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Haberman_KFold.py
1. load_data:
- input: ()
- output: (X, y) đọc dataset haberman, chuyển nhãn 2 → 1, 1 → -1, trả về toàn bộ X, y (dùng cho KFold)
### Haberman_CNN.py
1. load_data:
- input: (test_size)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset haberman, chuyển nhãn 2 → 1, 1 → 0, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Haberman_CNN_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset haberman, chuyển nhãn 2 → 1, 1 → 0, thay đổi tỉ lệ mất cân bằng bằng change_rate_data_cnn, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Haberman_TestSize.py
1. load_data:
- input: (test_size)
- output: (X_train, y_train, X_test, y_test) đọc dataset haberman, chuyển nhãn 2 → 1, 1 → -1, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Haberman_TestSize_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset haberman, chuyển nhãn 2 → 1, 1 → -1, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Pima.py
1. load_data:
- input: ()
- output: (X, y) đọc dataset diabetes (Pima Indians), chuyển nhãn 1 → 1, 0 → -1, trả về toàn bộ X, y (dùng cho KFold)
### Pima_CNN.py
1. load_data:
- input: (test_size)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset diabetes (Pima Indians), giữ nguyên nhãn (1 → 1, 0 → 0), chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Pima_CNN_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset diabetes (Pima Indians), giữ nguyên nhãn (1 → 1, 0 → 0), thay đổi tỉ lệ mất cân bằng bằng change_rate_data_cnn, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Pima_TestSize.py
1. load_data:
- input: (test_size)
- output: (X_train, y_train, X_test, y_test) đọc dataset diabetes (Pima Indians), chuyển nhãn 1 → 1, 0 → -1, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Pima_TestSize_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset diabetes (Pima Indians), chuyển nhãn 1 → 1, 0 → -1, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Transfution_Kfold.py
1. load_data:
- input: ()
- output: (X, y) đọc dataset transfusion, chuyển nhãn 1 → 1, 0 → -1, trả về toàn bộ X, y (dùng cho KFold)
### Transfution_CNN.py
1. load_data:
- input: (test_size)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset transfusion, giữ nguyên nhãn (1 → 1, 0 → 0), chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Transfution_CNN_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset transfusion, giữ nguyên nhãn (1 → 1, 0 → 0), thay đổi tỉ lệ mất cân bằng bằng change_rate_data_cnn, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Transfution_TestSize.py
1. load_data:
- input: (test_size)
- output: (X_train, y_train, X_test, y_test) đọc dataset transfusion, chuyển nhãn 1 → 1, 0 → -1, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Transfution_TestSize_IR.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset transfusion, chuyển nhãn 1 → 1, 0 → -1, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Yeast_CNN.py
1. load_data:
- input: (test_size)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset yeast, chuyển nhãn ME2 → 1, các lớp khác → 0, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Yeast_TestSize.py
1. load_data:
- input: (test_size)
- output: (X_train, y_train, X_test, y_test) đọc dataset yeast, chuyển nhãn ME2 → 1, các lớp khác → -1, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler
### Satimage_CNN.py
1. load_data:
- input: (test_size)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset satimage_full, chuyển nhãn class 4 → 1, các lớp khác → 0, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### Page_blocks_CNN.py
1. load_data:
- input: (test_size)
- output: (X_train, train_labels, X_test, test_labels) đọc dataset page-blocks, chuyển nhãn class 5 → 1, các lớp khác → 0, chia train/test theo test_size (stratify), encode nhãn bằng to_categorical
### churn.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset churn, chuyển nhãn False. → -1, True. → 1, encode các cột categorical (State, Phone, Int'l Plan, VMail Plan, VMail Message) bằng LabelEncoder, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler, giảm chiều bằng PCA(n_components=15)
### churn1.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset churn, chuyển nhãn False. → -1, True. → 1, encode các cột categorical bằng LabelEncoder, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test cố định test_size=0.25, chuẩn hóa X bằng StandardScaler, giảm chiều bằng PCA(n_components=15)
### indian_liver_patient.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset Indian Liver Patient, chuyển Gender (Female → 0, Male → 1), chuyển nhãn Dataset (1 → -1, 2 → 1), xử lý missing values bằng SimpleImputer(mean), encode cột bằng LabelEncoder, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler, giảm chiều bằng PCA(n_components=6)
### seismic_bumps.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset seismic bumps, chuyển nhãn 1 → 1, 0 → -1, encode các cột categorical (cột 0, 1, 2, 7) bằng LabelEncoder, thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify), chuẩn hóa X bằng StandardScaler, giảm chiều bằng PCA(n_components=11)
### spect_heart.py
1. load_data:
- input: (test_size, new_rate)
- output: (X_train, y_train, X_test, y_test) đọc dataset SPECT Heart, chuyển nhãn 1 → -1, 0 → 1 (đảo ngược), thay đổi tỉ lệ mất cân bằng bằng change_rate_data, chia train/test theo test_size (stratify)
# fuzzy
## weight
### fuzzy.py
#### method
1. own_class_center:
- input: (X, y)
- output: (d_cen) Tính tâm (mean) của hai lớp positive và negative. Sau đó trừ mỗi điểm dữ liệu cho tâm của lớp mà nó thuộc về, rồi tính norm (Euclidean distance) theo từng hàng. Kết quả trả về vector d_cen chứa khoảng cách từ mỗi điểm dữ liệu tới tâm lớp của nó.
2. own_class_center_divided
- input: (X, y)
- output: (d_divided) Tính khoảng cách từ mỗi điểm tới tâm lớp của nó và tới tâm lớp đối diện, sau đó trả về tỉ lệ 3 * (distance_to_own_center / distance_to_opposite_center).
3. own_class_center_opposite
- input: (X, y)
- output: (d_cen) Tính tâm của hai lớp positive và negative. Sau đó trừ mỗi điểm dữ liệu cho tâm của lớp đối diện (positive trừ tâm negative và ngược lại). Tiếp theo tính norm (Euclidean distance) theo từng hàng và lấy nghịch đảo của khoảng cách. Kết quả trả về vector thể hiện mức độ gần của mỗi điểm tới tâm lớp đối diện.
4. distance_center_own_opposite_tam
- input: (X, y)
- output: (d_cen_own, d_cen_opposite, d_tam) Tính khoảng cách từ mỗi điểm dữ liệu tới tâm lớp của nó và tâm lớp đối diện bằng norm theo từng hàng. Đồng thời tính khoảng cách giữa hai tâm lớp (center positive và center negative). Hàm trả về ba giá trị: khoảng cách tới tâm lớp của chính nó, khoảng cách tới tâm lớp đối diện và khoảng cách giữa hai tâm lớp.
5. estimated_hyper_lin
- input: (X, y)
- output: (d_cen) Ước lượng khoảng cách của mỗi điểm dữ liệu tới siêu phẳng tuyến tính giả định bằng cách tính tâm của toàn bộ dữ liệu, sau đó trừ các điểm dữ liệu cho tâm này và tính norm theo từng hàng. Kết quả trả về vector khoảng cách của các điểm tới tâm dữ liệu.
6. actual_hyper_lin
- input: (X, y, kernel, C, gamma)
- output: (d) Huấn luyện mô hình SVM với dữ liệu đầu vào (X, y) để tìm siêu phẳng phân tách. Sau đó sử dụng decision_function để tính khoảng cách có hướng từ mỗi điểm dữ liệu tới siêu phẳng phân tách thực tế. Kết quả được nhân với nhãn y để đảm bảo giá trị dương cho các điểm được phân loại đúng.
#### function
1. lin
- input: (d, delta)
- output: (f) Chuẩn hóa vector khoảng cách d theo dạng tuyến tính. Giá trị được tính bằng 1 - d / max(d) để đưa khoảng cách về khoảng [0,1], trong đó các điểm gần hơn sẽ có giá trị lớn hơn. delta được thêm vào để tránh chia cho 0.
2. lin_center_own
- input: (d, pos_ind, neg_ind, delta)
- output: (f) Chuẩn hóa khoảng cách tuyến tính riêng cho từng lớp. Khoảng cách của các điểm thuộc lớp positive được chuẩn hóa theo giá trị lớn nhất của lớp positive, và tương tự cho lớp negative. Điều này giúp cân bằng ảnh hưởng giữa hai lớp.
3. exp
- input: (d, beta)
- output: (f) Chuyển đổi khoảng cách d sang giá trị trọng số bằng hàm logistic dạng mũ. Tham số beta điều chỉnh tốc độ suy giảm của hàm. Khoảng cách lớn sẽ cho giá trị nhỏ hơn, giúp giảm ảnh hưởng của các điểm xa.
4. gau
- input: (d, u, sigma)
- output: (f) Áp dụng hàm Gaussian để chuyển đổi khoảng cách d thành trọng số. Giá trị trọng số giảm theo phân phối Gaussian quanh trung tâm u, với độ rộng điều chỉnh bởi tham số sigma.
5. func_own_opp
- input: (d_cenpos, d_cenneg, pos_ind, neg_ind, d_tam)
- output: (f) Tính trọng số fuzzy dựa trên khoảng cách tới tâm lớp của chính nó và tâm lớp đối diện. Công thức kết hợp hai khoảng cách này cùng với khoảng cách giữa hai tâm lớp để đánh giá vị trí tương đối của điểm dữ liệu trong không gian hai lớp.
6. func_own_opp_new
- input: (d_cenpos, d_cenneg, pos_ind, neg_ind, d_tam, delta)
- output: (f) Biến thể cải tiến của hàm func_own_opp. Trọng số được tính bằng tỉ lệ giữa khoảng cách tới tâm lớp đối diện và tổng khoảng cách tới tâm lớp của nó cộng với khoảng cách giữa hai tâm lớp. delta được thêm vào để tránh chia cho 0.
7. func_own_opp_new_v1
- input: (d_cenpos, d_cenneg, pos_ind, neg_ind, d_tam, delta)
- output: (f) Phiên bản cải tiến thứ nhất của hàm fuzzy dựa trên khoảng cách tới tâm lớp. Hàm sử dụng giá trị lớn nhất và nhỏ nhất của khoảng cách trong từng lớp để chuẩn hóa và điều chỉnh trọng số của từng điểm dữ liệu.
8. func_own_opp_new_v2
- input: (d_cenpos, d_cenneg, pos_ind, neg_ind, d_tam, delta)
- output: (f) Phiên bản cải tiến thứ hai của hàm fuzzy. Thay vì dùng tỉ lệ khoảng cách, hàm sử dụng tổng khoảng cách tới hai tâm lớp kết hợp với các giá trị cực trị của khoảng cách trong từng lớp để chuẩn hóa và tính trọng số fuzzy cho mỗi điểm dữ liệu.
## model.py
### class SVC
- init: (X, y, kernel_name, C, gamma, r, d, distribution_weight) khởi tạo SVM classifier hỗ trợ nhiều kernel (linear, rbf, polynomial) với các tham số tương ứng
1. fit:
- input: (X, y)
- output: không trả về, tính Lagrange multipliers bằng solver, tìm support vectors (λ ≥ 0.01) và margin vectors (0 < λ < C), tính weight w (nếu kernel linear) và bias b, lưu vào thuộc tính self
2. compute_bias:
- input: (X, y)
- output: (b) tính bias. Nếu kernel linear: b = mean(y[S] - X[S]·w). Nếu kernel khác: tính trung bình trên các margin vectors dùng kernel function
3. compute_weight:
- input: (X, y)
- output: (w) tính weight vector w = X^T · (y * λ), chỉ dùng cho kernel linear
4. decision_function:
- input: (X)
- output: (giá trị quyết định) nếu kernel linear: X·w + b. Nếu kernel khác: tính tổng λ_s * y_s * K(X_sv, X_i) trên các support vectors, cộng b
5. predict:
- input: (X)
- output: (nhãn dự đoán) trả về sign(decision_function(X))
6. find_support_vectors (static):
- input: (lam)
- output: (index) tìm các chỉ số có λ ≥ 0.01
7. find_margin_vertors (static):
- input: (lam, C)
- output: (index) tìm các chỉ số có 0 < λ < C (các điểm nằm trên margin)

# svm
## methods.py
1. dual_problem_quadratic_program:
- input: (X, y, C, distribution_weights)
- output: (P, q, G, h, A, b) xây dựng các ma trận cho bài toán Quadratic Programming dạng dual của SVM. Nếu C=None → hard-SVM (chỉ ràng buộc λ ≥ 0). Nếu C có giá trị → soft-SVM (ràng buộc 0 ≤ λ ≤ C*W). distribution_weights dùng để nhân vào ràng buộc trên của λ
2. dual_problem_quadratic_solver:
- input: (P, q, G, h, A, b)
- output: (solution) giải bài toán QP bằng cvxopt.solvers.qp
3. svm_lagrange_mutipliers:
- input: (solution)
- output: (λ) trích xuất vector Lagrange multipliers từ kết quả solver, shape (N, 1)
4. svm_support_vectors:
- input: (lamda)
- output: (index) tìm các chỉ số có λ ≥ 0.01 (support vectors)
5. svm_weight:
- input: (X, y, lamda)
- output: (w) tính weight vector w = X^T · (y * λ), shape (d,)
6. svm_bias:
- input: (X, y, S, weight)
- output: (b) tính bias b = mean(y[S] - X[S]·w) trên tập support vectors S
7. svm_pred:
- input: (X, w, b)
- output: (pred) dự đoán nhãn bằng sign(X·w + b)
8. svm_accuracy:
- input: (pred, y)
- output: (accuracy) tính tỉ lệ dự đoán đúng mean(y == pred)
## application.py
1. fit:
- input: (X, y, C, distribution_weight)
- output: (w, b) quy trình hoàn chỉnh: xây dựng QP → giải QP → tính Lagrange → tìm support vectors → tính weight w → tính bias b

# wsvm
## methods.py
1. dual_problem_quadratic_program:
- input: (X, y, C, distribution_weights)
- output: (P, q, G, h, A, b) tương tự svm/methods.py, xây dựng các ma trận QP cho Weighted SVM với distribution_weights nhân vào ràng buộc trên của λ
2. dual_problem_quadratic_solver:
- input: (P, q, G, h, A, b)
- output: (solution) giải QP bằng cvxopt.solvers.qp
3. svm_lagrange_mutipliers:
- input: (solution)
- output: (λ) trích xuất vector Lagrange multipliers
4. svm_support_vectors:
- input: (lamda)
- output: (index) tìm chỉ số có λ ≥ 0.01
5. svm_weight:
- input: (X, y, lamda)
- output: (w) tính weight vector w = X^T · (y * λ)
6. svm_bias:
- input: (X, y, S, weight)
- output: (b) tính bias trên tập support vectors
7. svm_pred:
- input: (X, w, b)
- output: (pred) dự đoán nhãn bằng sign(X·w + b)
8. svm_accuracy:
- input: (pred, y)
- output: (accuracy) tính tỉ lệ dự đoán đúng
## application.py
### class Wsvm
- init: (C, distribution_weight) khởi tạo Weighted SVM với hệ số C và trọng số phân bố
1. fit:
- input: (X, y)
- output: không trả về, giải QP với distribution_weight → tính Lagrange → tìm support vectors → tính w, b, lưu vào self.w và self.b
2. predict:
- input: (X)
- output: (H) dự đoán nhãn bằng sign(X·w + b)

# methods.py
1. intinitialization_weight_adjustment:
- input: (X, y, proposed, theta)
- output: (w) khởi tạo trọng số ban đầu cho AdaBoost. Nếu proposed=False: w = 1/N đều nhau. Nếu proposed=True: tính eps = N_pos/N_neg, delta_pos = (1-eps)^theta/(eps*N), delta_neg = (1-eps)^theta/N, sau đó w[pos] = 1/N + delta_pos, w[neg] = 1/N - delta_neg (thiên vị cho lớp thiểu số)
2. intinitialization_instance_categorization:
- input: (N)
- output: vector ones(N), khởi tạo trọng số instance categorization đều bằng 1
3. predict_svm:
- input: (X, w, b)
- output: (pred) dự đoán nhãn bằng sign(X·w + b)
4. find_true_false_index:
- input: (y, pred)
- output: (true_index, false_index, false_index_P, false_index_N) tìm chỉ số dự đoán đúng, sai, sai trên lớp positive (+1) và sai trên lớp negative (-1)
5. confident:
- input: (W, false_index_P, false_index_N, proposed_alpha)
- output: (alpha, eps) tính confident (trọng số) của weak learner. Nếu proposed_alpha=False: eps = tổng_W_sai / tổng_W, alpha = 1/2 * ln((1-eps)/eps). Nếu proposed_alpha=True: eps = esp_N + esp_P*(1-(esp_N+esp_P)), tăng penalty cho lỗi phân loại sai lớp positive
6. update_weight_adjustment:
- input: (W, alpha, true_index, false_index)
- output: (W) cập nhật trọng số AdaBoost: W[đúng] *= exp(-alpha), W[sai] *= exp(alpha), rồi chuẩn hóa W = W / sum(W)
7. update_weights:
- input: (weights, y_pred, y, alpha)
- output: (weights) phiên bản cập nhật trọng số khác: nhân exp(-alpha * indicator), rồi chuẩn hóa
8. update_instance_categorization_final:
- input: (X, y, w, b)
- output: (C) phân loại mẫu dựa trên B = X·w + b. BSV (-1 < B < 1): trọng số = num_BSV / (2 * num_pos_hoặc_neg_BSV). SV (B = ±1): trọng số = num_SV / (2 * num_pos_hoặc_neg_SV). Noise (B > 1, y = -1): trọng số = exp(num_noise / num_positive)
9. update_instance_categorization:
- input: (X, y, w, b)
- output: (C) phân loại mẫu dựa trên A = 1 - y*(X·w+b). BSV (0 < A < 2): trọng số = num_BSV / (2 * num_pos_hoặc_neg_BSV). SV (A = 0): trọng số = num_SV / (2 * num_pos_hoặc_neg_SV). Positive noise (A ≤ 2, y = 1): trọng số = exp(num_noise / num_positive)

# trainning_of_adaboost.py
1. fit:
- input: (X, y, M, C, instance_categorization, proposed_preprocessing, proposed_alpha, test_something, theta)
- output: (w, b, alpha) huấn luyện Im.AdaBoost với SVM tự viết. Vòng lặp M lần: train SVM → predict → tìm true/false index → tính alpha → cập nhật trọng số W_ada. Nếu instance_categorization=True: nhân W_ada với B_ada (instance categorization) và cập nhật B_ada bằng update_instance_categorization_final sau mỗi vòng
2. predict:
- input: (X, w, b, alpha, M)
- output: (pred) dự đoán bằng H(x) = sign(Σ alpha_i * (X·w_i + b_i))

# adaboost_svm.py
1. fit:
- input: (X, y, M, C, instance_categorization, proposed, theta)
- output: (w, b, alpha) huấn luyện AdaBoost với sklearn.svm.SVC(kernel='linear', C=10000). Nếu instance_categorization=True: dùng class_weight = W_ada * C_ada. Nếu False: truyền W_ada làm sample_weight cho SVC.fit. Cập nhật trọng số và instance categorization sau mỗi vòng
2. predict:
- input: (X, w, b, alpha, M)
- output: (pred) dự đoán bằng H(x) = sign(Σ alpha_i * (X·w_i + b_i))

# ImAda_DecisionTree.py
1. fit:
- input: (X, y, M, proposed_preprocessing, proposed_alpha, theta)
- output: (clfs, alpha) huấn luyện Im.AdaBoost với DecisionTreeClassifier làm weak learner. Vòng lặp M lần: train cây quyết định với sample_weight=W_ada → predict → tính alpha → cập nhật trọng số. Trả về danh sách các cây và alpha tương ứng
2. predict:
- input: (X, alpha, clfs)
- output: (y_pred) dự đoán ensemble: y = sign(Σ alpha_i * h_i(X)), trong đó h_i(X) được chuyển về {-1, 1}

# report.py
1. report:
- input: (Label, pred)
- output: (DataFrame) tính precision, recall, fscore, support cho từng lớp {-1, 1} bằng sklearn, trả về pandas DataFrame

# Find_C.py
- Script tìm giá trị C tối ưu cho SVM. Duyệt C từ 50 đến 20000 (bước 10), với nhiều test_size và imbalance rate. Mỗi cấu hình chạy 10 lần, chọn C cho F-score cao nhất, rồi lấy trung bình C. Ghi kết quả ra file text.

# save_alpha.py
- Script chạy thí nghiệm và lưu chuỗi giá trị alpha qua các vòng lặp AdaBoost cho các biến thể (WSVM, Nova1+WSVM, Nova1+2+WSVM) để phân tích hành vi hội tụ. Ghi alpha ra file text.

# adaboosttree.py
- Script chạy baseline AdaBoost chuẩn (sklearn.AdaBoostClassifier, n_estimators=100) trên tập Co-Author. Duyệt nhiều theta, imbalance rate, test_size. Chạy nhiều lần lấy trung bình precision, recall, F-score. Ghi kết quả ra file text.

# adaboostWSVM.py
- Script chạy thí nghiệm AdaBoost + WSVM và Nova1+SVM trên tập Indian Liver Patient. Duyệt nhiều theta, imbalance rate, test_size. Chạy nhiều lần lấy trung bình metric. Ghi kết quả ra file text.

# __init__.py
- File khởi tạo package gốc. Import và export các hàm chính từ methods.py, trainning_of_adaboost.py, adaboost_svm.py và report.py.

# __main__.py
- Script thí nghiệm chính. Chạy so sánh các biến thể: WSVM (bài báo 2016), Nova1+WSVM (proposed preprocessing). Cấu hình: M=10, C=10000, theta=0.2. Duyệt imbalance rate và test_size, chạy nhiều lần lấy trung bình. Dataset: Vertebral Column. Ghi kết quả ra file text.

# __main1__.py
- Script thí nghiệm so sánh đầy đủ 8 biến thể: ADA+SVM, ADA+WSVM, Nova1+SVM, Nova1+WSVM, Nova2+SVM, Nova2+WSVM, Nova1+2+SVM, Nova1+2+WSVM. Dataset: Churn. Ghi kết quả ra file text.

# __main2__.py
- Script thí nghiệm tương tự __main__.py nhưng thêm tính confusion matrix cho mỗi biến thể (WSVM, Nova1+WSVM, Nova1+2+WSVM). Dataset: Vertebral Column. Ghi kết quả ra file text.
