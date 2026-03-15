# Im.AdaBoost — An Adaptive Fuzzy Weight Algorithm for the Class Imbalance Learning Problem

---

## 1. Tổng quan dự án

### Mục tiêu
Dự án nghiên cứu và triển khai thuật toán **Im.AdaBoost** (Imbalanced AdaBoost) — một biến thể cải tiến của AdaBoost dành riêng cho bài toán **phân loại dữ liệu mất cân bằng lớp (class imbalance)**.

### Bài toán nghiên cứu
Trong thực tế, nhiều tập dữ liệu có số lượng mẫu của lớp thiểu số (minority/positive class) ít hơn rất nhiều so với lớp đa số (majority/negative class). Các thuật toán phân loại truyền thống thường bị lệch về lớp đa số, dẫn đến hiệu suất thấp trên lớp thiểu số. Dự án này đề xuất các cơ chế cải tiến AdaBoost kết hợp SVM để giải quyết vấn đề này.

### Phương pháp / Thuật toán chính
1. **AdaBoost + SVM**: Sử dụng SVM (Support Vector Machine) làm weak learner cho AdaBoost thay vì Decision Tree truyền thống.
2. **Weighted SVM (WSVM)**: SVM có trọng số dựa trên instance categorization (phân loại mẫu theo BSV, SV, Noise) — dựa trên bài báo 2016.
3. **Proposed Preprocessing (Nova 1)**: Khởi tạo trọng số ban đầu (weight adjustment) có thiên vị cho lớp thiểu số dựa trên tham số theta và tỷ lệ mất cân bằng epsilon.
4. **Proposed Alpha (Nova 2)**: Cải tiến công thức tính confident (alpha) trong AdaBoost, tăng penalty cho lỗi phân loại sai lớp thiểu số.
5. **Fuzzy Weight**: Tính trọng số mờ (fuzzy membership) dựa trên khoảng cách đến trung tâm lớp (class center).
6. **CNN (Convolutional Neural Network)**: Thử nghiệm mạng CNN trên một số tập dữ liệu.
7. **AdaBoost + Decision Tree**: So sánh baseline với AdaBoost chuẩn sử dụng Decision Tree (sklearn).

### Các thành phần chính
- Module SVM tự viết (giải bài toán Quadratic Programming qua `cvxopt`)
- Module WSVM (Weighted SVM) tự viết
- Module Fuzzy weight
- Thuật toán AdaBoost cải tiến (`trainning_of_adaboost.py`)
- Hệ thống load và xử lý dữ liệu cho nhiều tập dữ liệu UCI
- Các script thí nghiệm và notebook phân tích

---

## 2. Kiến trúc tổng thể hệ thống

### Pipeline chính

```
Input Dataset (CSV)
     │
     ▼
Data Loader (data/*.py)
  ├── Đọc CSV
  ├── Mã hóa nhãn → {-1, +1}
  ├── Thay đổi tỷ lệ mất cân bằng (change_rate_data)
  ├── Chia train/test (train_test_split)
  └── Chuẩn hóa (StandardScaler, PCA)
     │
     ▼
Training (trainning_of_adaboost.py / adaboost_svm.py)
  ├── Khởi tạo trọng số W_ada (methods.py)
  ├── Vòng lặp AdaBoost M lần:
  │   ├── Train SVM/WSVM weak learner (svm/ hoặc wsvm/)
  │   ├── Predict trên tập train
  │   ├── Tính alpha (confident)
  │   ├── Cập nhật trọng số (weight adjustment)
  │   └── Cập nhật instance categorization (nếu bật)
  └── Trả về danh sách (w, b, alpha)
     │
     ▼
Prediction
  H(x) = sign( Σ alpha_i * (x·w_i + b_i) )
     │
     ▼
Evaluation (precision, recall, F-score)
  └── Ghi kết quả ra file .txt hoặc .csv
```

### Các module chính và tương tác

| Module | Chức năng | Phụ thuộc |
|--------|-----------|-----------|
| `methods.py` | Các hàm cốt lõi của AdaBoost (khởi tạo trọng số, tính alpha, cập nhật trọng số, instance categorization) | numpy |
| `trainning_of_adaboost.py` | Hàm `fit()` và `predict()` chính của Im.AdaBoost | methods, svm |
| `adaboost_svm.py` | Phiên bản AdaBoost + SVM sử dụng sklearn SVC | methods, sklearn.svm |
| `svm/` | SVM tự triển khai (giải QP bằng cvxopt) | cvxopt, numpy |
| `wsvm/` | Weighted SVM tự triển khai | cvxopt, numpy |
| `fuzzy/` | Fuzzy membership weight | sklearn.svm, numpy |
| `data/` | Data loader cho từng tập dữ liệu | pandas, sklearn |
| `common/` | Hàm thay đổi tỷ lệ dữ liệu mất cân bằng | numpy |
| `__main__.py`, `__main1__.py`, `__main2__.py` | Script chạy thí nghiệm | tất cả module trên |

---

## 3. Sơ đồ cấu trúc Repository

```
Im.AdaBoost-master/
│
├── __init__.py                    # Export các hàm chính của dự án
├── __main__.py                    # Script thí nghiệm chính (nhiều biến thể AdaBoost)
├── __main1__.py                   # Script thí nghiệm phiên bản 1 (Churn dataset)
├── __main2__.py                   # Script thí nghiệm phiên bản 2 (Vertebral + confusion matrix)
├── methods.py                     # Hàm core: weight init, alpha, instance categorization
├── trainning_of_adaboost.py       # Thuật toán Im.AdaBoost (fit/predict) — SVM tự viết
├── adaboost_svm.py                # AdaBoost + SVM (dùng sklearn SVC)
├── adaboostWSVM.py                # Script chạy AdaBoost + WSVM trên Indian Liver Patient
├── adaboosttree.py                # Script chạy AdaBoost + Decision Tree (sklearn)
├── ImAda_DecisionTree.py          # Module Im.AdaBoost dùng Decision Tree làm weak learner
├── Find_C.py                      # Tìm giá trị C tối ưu cho SVM
├── report.py                      # Tính precision, recall, F-score
├── save_alpha.py                  # Script lưu và phân tích giá trị alpha
├── README.md                      # File README hiện tại
│
├── FullCode_ImADA.ipynb           # Notebook chính: toàn bộ pipeline Im.AdaBoost
├── FullCode_ImADA_ver1.ipynb      # Notebook phiên bản 1
├── FullCode_ImADA_ver2.ipynb      # Notebook phiên bản 2
├── FullCode_ImADA_UCI.ipynb       # Notebook thí nghiệm trên nhiều tập UCI
├── Train_CNN.ipynb                # Notebook huấn luyện CNN
│
├── svm/                           # Module SVM tự triển khai
│   ├── __init__.py                # Export: fit, QP solver, weight, bias, ...
│   ├── methods.py                 # Giải Quadratic Programming (cvxopt), tính w, b
│   └── application.py             # Hàm fit() tổng hợp: QP → Lagrange → SV → w, b
│
├── wsvm/                          # Module Weighted SVM tự triển khai
│   ├── __init__.py                # (chứa code cũ, hiện bị comment)
│   ├── methods.py                 # Giải QP có trọng số phân bố
│   └── application.py             # Class Wsvm: fit(), predict()
│
├── fuzzy/                         # Module Fuzzy Weight
│   ├── model.py                   # Class SVC (fuzzy SVM) với nhiều kernel
│   └── weight/
│       └── fuzzy.py               # Tính fuzzy membership: khoảng cách đến class center
│
├── common/                        # Tiện ích chung
│   └── change_rate_data.py        # Hàm thay đổi tỷ lệ mất cân bằng
│
├── data/                          # Module load và xử lý dữ liệu
│   ├── __init__.py                # Import các data loader
│   ├── common/
│   │   ├── __init__.py
│   │   └── change_rate_data.py    # change_rate_data(), change_rate_data_cnn()
│   ├── datasets/                  # Thư mục chứa file CSV dữ liệu gốc
│   │   ├── common/
│   │   │   ├── __init__.py
│   │   │   └── change_rate_data.py
│   │   ├── Vertebral_column.csv
│   │   ├── ecoli_new.csv
│   │   ├── haberman.csv
│   │   ├── diabetes.csv
│   │   ├── churn.csv
│   │   ├── transfusion.csv
│   │   ├── yeast.csv
│   │   ├── Spect_Heart.csv
│   │   ├── seismic-bumps.csv
│   │   ├── Co_Author_100_500_1.csv
│   │   ├── CoAuthor_500_2500.csv
│   │   ├── ... (nhiều biến thể Co-Author)
│   │   └── abalone.csv
│   │
│   ├── Vertebral_column.py        # Loader: Vertebral Column (test_size split)
│   ├── Vertebral_column_KF.py     # Loader: Vertebral Column (KFold)
│   ├── Co_Author.py               # Loader: Co-Author (test_size split)
│   ├── Co_Author_KF.py            # Loader: Co-Author (KFold)
│   ├── Co_Author_CNN.py           # Loader: Co-Author cho CNN
│   ├── Co_Author_CNN_IR.py        # Loader: Co-Author CNN + Imbalance Ratio
│   ├── Co_Author_TestSize.py      # Loader: Co-Author TestSize
│   ├── Co_Author_TestSize_IR.py   # Loader: Co-Author TestSize + Imbalance Ratio
│   ├── Ecoli_Kfold.py             # Loader: Ecoli (KFold)
│   ├── Ecoli_CNN.py               # Loader: Ecoli cho CNN
│   ├── Ecoli_CNN_IR.py            # Loader: Ecoli CNN + IR
│   ├── Ecoli_TestSize.py          # Loader: Ecoli (TestSize split)
│   ├── Ecoli_TestSize_IR.py       # Loader: Ecoli TestSize + IR
│   ├── Haberman_KFold.py          # Loader: Haberman (KFold)
│   ├── Haberman_CNN.py            # Loader: Haberman cho CNN
│   ├── Haberman_CNN_IR.py         # Loader: Haberman CNN + IR
│   ├── Haberman_TestSize.py       # Loader: Haberman (TestSize split)
│   ├── Haberman_TestSize_IR.py    # Loader: Haberman TestSize + IR
│   ├── Pima.py                    # Loader: Pima Indians Diabetes (KFold)
│   ├── Pima_CNN.py                # Loader: Pima cho CNN
│   ├── Pima_CNN_IR.py             # Loader: Pima CNN + IR
│   ├── Pima_TestSize.py           # Loader: Pima (TestSize)
│   ├── Pima_TestSize_IR.py        # Loader: Pima TestSize + IR
│   ├── Transfution_Kfold.py       # Loader: Blood Transfusion (KFold)
│   ├── Transfution_CNN.py         # Loader: Transfusion cho CNN
│   ├── Transfution_CNN_IR.py      # Loader: Transfusion CNN + IR
│   ├── Transfution_TestSize.py    # Loader: Transfusion (TestSize)
│   ├── Transfution_TestSize_IR.py # Loader: Transfusion TestSize + IR
│   ├── Yeast_CNN.py               # Loader: Yeast cho CNN
│   ├── Yeast_TestSize.py          # Loader: Yeast (TestSize)
│   ├── Satimage_CNN.py            # Loader: Satimage cho CNN
│   ├── Page_blocks_CNN.py         # Loader: Page Blocks cho CNN
│   ├── Abanole_CNN.py             # Loader: Abalone cho CNN
│   ├── Abanole_TestSize.py        # Loader: Abalone (TestSize)
│   ├── churn.py                   # Loader: Churn
│   ├── churn1.py                  # Loader: Churn (biến thể)
│   ├── indian_liver_patient.py    # Loader: Indian Liver Patient
│   ├── indian_liver_patient.csv   # Dữ liệu CSV gốc (bản sao)
│   ├── seismic_bumps.py           # Loader: Seismic Bumps
│   ├── seismic_bumps.csv          # Dữ liệu CSV gốc (bản sao)
│   └── spect_heart.py             # Loader: SPECT Heart
│
├── Experiment/                    # Kết quả thí nghiệm (CSV)
│   ├── Data_Co_Author_*.csv       # Kết quả trên Co-Author dataset
│   ├── Data_Ecoli_*.csv           # Kết quả trên Ecoli dataset
│   ├── Data_Haberman_*.csv        # Kết quả trên Haberman dataset
│   ├── Data_Pima_*.csv            # Kết quả trên Pima dataset
│   ├── Data_Transfution_*.csv     # Kết quả trên Transfusion dataset
│   └── Data_Vertebral_column_*.csv# Kết quả trên Vertebral Column dataset
│
└── Model/                         # Lưu model đã train
    └── Ecoli_Kfold_*.h5           # Model Keras đã train (Ecoli)
```

---

## 4. Giải thích chi tiết từng folder

### `svm/` — Module SVM tự triển khai
- **Mục đích**: Triển khai SVM từ đầu bằng cách giải bài toán Quadratic Programming (QP) sử dụng thư viện `cvxopt`.
- **Vai trò**: Là weak learner chính cho thuật toán AdaBoost. Hỗ trợ cả hard-SVM và soft-SVM, có thể nhận distribution weight từ AdaBoost.
- **Liên hệ**: Được gọi bởi `trainning_of_adaboost.py` thông qua `svm.fit()`.

### `wsvm/` — Module Weighted SVM
- **Mục đích**: Triển khai Weighted SVM, cho phép mỗi mẫu dữ liệu có trọng số riêng (distribution weight) trong ràng buộc SVM.
- **Vai trò**: Kết hợp với instance categorization để gán trọng số khác nhau cho BSV, SV, và noise.
- **Liên hệ**: Có thể được gọi thay cho `svm/` khi cần phân loại có trọng số.

### `fuzzy/` — Module Fuzzy Weight
- **Mục đích**: Tính trọng số mờ (fuzzy membership) cho từng mẫu dữ liệu dựa trên khoảng cách đến trung tâm lớp.
- **Vai trò**: Cung cấp class `SVC` mở rộng hỗ trợ nhiều kernel (linear, RBF, polynomial) và phương pháp tính fuzzy weight qua khoảng cách (own class center, opposite class center, estimated/actual hyperplane).
- **Liên hệ**: Được sử dụng trong các notebook thí nghiệm.

### `data/` — Module dữ liệu
- **Mục đích**: Chứa tất cả data loader cho các tập dữ liệu UCI và Co-Author. Mỗi file loader đọc CSV, mã hóa nhãn, thay đổi tỷ lệ mất cân bằng, chia tập train/test, chuẩn hóa dữ liệu.
- **Vai trò**: Điểm đầu vào dữ liệu cho mọi thí nghiệm.
- **Dữ liệu**: Folder `datasets/` chứa các file CSV gốc (Vertebral Column, Ecoli, Haberman, Diabetes/Pima, Churn, Transfusion, Yeast, SPECT Heart, Seismic Bumps, Co-Author, Abalone, v.v.)
- **Biến thể loader**:
  - `*_KFold.py` / `*_KF.py`: Trả về X, y cho KFold cross-validation
  - `*_TestSize.py`: Chia train/test theo test_size
  - `*_TestSize_IR.py`: Test size + thay đổi Imbalance Ratio
  - `*_CNN.py`: Chuẩn bị dữ liệu cho CNN (one-hot encode labels)
  - `*_CNN_IR.py`: CNN + thay đổi Imbalance Ratio

### `data/common/` — Tiện ích dữ liệu
- **Mục đích**: Chứa hàm `change_rate_data()` dùng để thay đổi tỷ lệ mất cân bằng bằng cách giảm số mẫu lớp thiểu số.
- **Liên hệ**: Được import bởi hầu hết data loader.

### `common/` — Tiện ích chung (root)
- **Mục đích**: Bản sao hàm `change_rate_data()` ở root level.
- **Liên hệ**: Có thể được import trực tiếp từ script chính.

### `Experiment/` — Kết quả thí nghiệm
- **Mục đích**: Lưu trữ kết quả chạy thí nghiệm dưới dạng CSV.
- **Dữ liệu**: File CSV chứa các metric (precision, recall, F-score) cho mỗi cấu hình (dataset, test_size, imbalance ratio, phương pháp).
- **Quy tắc đặt tên**: `Data_{Dataset}_{Config}_{DateTime}_{EvalMethod}.csv`

### `Model/` — Model đã huấn luyện
- **Mục đích**: Lưu model Keras/TensorFlow đã train (file `.h5`).
- **Dữ liệu**: Hiện có model Ecoli KFold.

---

## 5. Giải thích từng file quan trọng

### `methods.py` — Core functions của Im.AdaBoost
- **Chức năng**: Chứa tất cả hàm cốt lõi cho thuật toán AdaBoost cải tiến.
- **Các hàm quan trọng**:
  - `intinitialization_weight_adjustment(X, y, proposed, theta)`: Khởi tạo trọng số W ban đầu. Nếu `proposed=True`, tính delta dựa trên tỷ lệ epsilon và theta để thiên vị cho lớp thiểu số.
  - `intinitialization_instance_categorization(N)`: Khởi tạo vector phân loại mẫu (tất cả = 1).
  - `predict_svm(X, w, b)`: Dự đoán bằng SVM: `sign(X·w + b)`.
  - `find_true_false_index(y, pred)`: Tìm index mẫu đúng/sai, tách riêng lỗi Positive (P) và Negative (N).
  - `confident(W, false_index_P, false_index_N, proposed_alpha)`: Tính alpha. Nếu `proposed_alpha=True`, tăng penalty cho lỗi phân loại sai lớp thiểu số: `eps = esp_N + esp_P*(1-(esp_N+esp_P))`.
  - `update_weight_adjustment(W, alpha, true_index, false_index)`: Cập nhật trọng số AdaBoost theo công thức chuẩn.
  - `update_instance_categorization(X, y, w, b)`: Phân loại mẫu thành BSV/SV/Noise dựa trên `A = 1 - y*(X·w+b)`, gán trọng số theo tỷ lệ lớp.
  - `update_instance_categorization_final(X, y, w, b)`: Phiên bản cải tiến với phân loại theo `B = X·w + b`.
- **Đầu vào**: Dữ liệu X, nhãn y, trọng số W, tham số mô hình (w, b).
- **Đầu ra**: Trọng số, alpha, vector phân loại mẫu.
- **Phụ thuộc**: numpy.
- **Được gọi bởi**: `trainning_of_adaboost.py`, `adaboost_svm.py`, `ImAda_DecisionTree.py`.

---

### `trainning_of_adaboost.py` — Thuật toán Im.AdaBoost chính
- **Chức năng**: Huấn luyện và dự đoán bằng Im.AdaBoost kết hợp SVM tự viết.
- **Các hàm quan trọng**:
  - `fit(X, y, M, C, instance_categorization, proposed_preprocessing, proposed_alpha, test_something, theta)`:
    - `instance_categorization=True`: Bật instance categorization (WSVM theo bài báo 2016).
    - `proposed_preprocessing=True`: Khởi tạo trọng số thiên vị cho thiểu số (Nova 1).
    - `proposed_alpha=True`: Dùng alpha cải tiến (Nova 2).
    - Trả về: `(w, b, alpha)` — danh sách weight, bias, confident cho M vòng.
  - `predict(X, w, b, alpha, M)`: `H(x) = sign(Σ alpha_i * (X·w_i + b_i))`.
- **Đầu vào**: X (features), y (labels), hyperparameters.
- **Đầu ra**: Model (w, b, alpha) hoặc predictions.
- **Phụ thuộc**: `methods.py`, `svm/`.
- **Được gọi bởi**: `__main__.py`, `__main1__.py`, `__main2__.py`, notebooks.

---

### `adaboost_svm.py` — AdaBoost + SVM (sklearn)
- **Chức năng**: Phiên bản AdaBoost sử dụng `sklearn.svm.SVC` thay vì SVM tự viết.
- **Hàm chính**: `fit()`, `predict()` tương tự `trainning_of_adaboost.py`.
- **Phụ thuộc**: `methods.py`, `sklearn.svm.SVC`.
- **Được gọi bởi**: `__main__.py`, `__main1__.py`.

---

### `ImAda_DecisionTree.py` — Im.AdaBoost + Decision Tree
- **Chức năng**: Phiên bản Im.AdaBoost sử dụng Decision Tree (`sklearn.tree.DecisionTreeClassifier`) làm weak learner.
- **Hàm chính**:
  - `fit(X, y, M, proposed_preprocessing, proposed_alpha, theta)`: Trả về `(clfs, alpha)`.
  - `predict(X, alpha, clfs)`: Dự đoán dựa trên ensemble các cây quyết định.
- **Phụ thuộc**: `methods.py`, `sklearn.tree`.

---

### `svm/application.py` — SVM fit
- **Chức năng**: Hàm `fit(X, y, C, distribution_weight)` tổng hợp quy trình SVM: giải QP → Lagrange → Support Vectors → tính w, b.
- **Đầu vào**: X, y, hệ số C, trọng số phân bố.
- **Đầu ra**: `(w, b)`.
- **Phụ thuộc**: `svm/methods.py`.

### `svm/methods.py` — SVM core (Quadratic Programming)
- **Chức năng**: Giải bài toán tối ưu SVM qua dạng dual (Quadratic Programming) sử dụng `cvxopt`.
- **Hàm chính**:
  - `dual_problem_quadratic_program(X, y, C, distribution_weights)`: Xây dựng ma trận P, q, G, h, A, b.
  - `dual_problem_quadratic_solver(P, q, G, h, A, b)`: Gọi `cvxopt.solvers.qp`.
  - `svm_lagrange_mutipliers(solution)`: Trích Lagrange multipliers.
  - `svm_support_vectors(lamda)`: Tìm support vectors (λ ≥ 0.01).
  - `svm_weight(X, y, lamda)`: Tính trọng số w.
  - `svm_bias(X, y, S, weight)`: Tính bias b.

---

### `wsvm/application.py` — Weighted SVM
- **Chức năng**: Class `Wsvm` với `fit()` và `predict()`, sử dụng distribution weight trong ràng buộc QP.
- **Phụ thuộc**: `wsvm/methods.py`.

---

### `fuzzy/model.py` — Fuzzy SVM
- **Chức năng**: Class `SVC` hỗ trợ nhiều kernel (linear, RBF, polynomial). Triển khai đầy đủ `fit()`, `predict()`, `decision_function()`, tìm support vectors và margin vectors.
- **Phụ thuộc**: `fuzzy/weight/fuzzy.py`, solver riêng.

### `fuzzy/weight/fuzzy.py` — Fuzzy membership functions
- **Chức năng**: Tính trọng số fuzzy cho mỗi mẫu dữ liệu.
- **Các phương pháp**:
  - `own_class_center(X, y)`: Khoảng cách đến trung tâm lớp mình.
  - `own_class_center_divided(X, y)`: Tỷ lệ khoảng cách own/opposite class.
  - `own_class_center_opposite(X, y)`: Nghịch đảo khoảng cách đến class đối diện.
  - `estimated_hyper_lin(X, y)`: Khoảng cách đến trung tâm toàn bộ dữ liệu.
  - `actual_hyper_lin(X, y)`: Khoảng cách đến hyperplane thực (dùng SVM).
  - `function.lin(d)`: Hàm chuyển khoảng cách → trọng số: `s = 1 - d/d_max`.

---

### `report.py` — Báo cáo metric
- **Chức năng**: Tính và trả về DataFrame chứa precision, recall, F-score cho từng lớp.
- **Đầu vào**: y_true, y_pred.
- **Đầu ra**: pandas DataFrame.
- **Phụ thuộc**: sklearn.metrics.
- **Được gọi bởi**: `__main__.py`, `__main1__.py`.

---

### `Find_C.py` — Tìm giá trị C tối ưu
- **Chức năng**: Duyệt qua nhiều giá trị C (từ 50 đến 20000) và test_size để tìm C cho F-score cao nhất.
- **Đầu vào**: Dataset Vertebral Column.
- **Đầu ra**: File text báo cáo C tối ưu cho từng cấu hình.
- **Phụ thuộc**: `svm/`, `data/Vertebral_column.py`.

---

### `save_alpha.py` — Lưu và phân tích giá trị alpha
- **Chức năng**: Chạy thí nghiệm và lưu chuỗi giá trị alpha qua các vòng lặp AdaBoost để phân tích hành vi hội tụ.
- **Phụ thuộc**: `trainning_of_adaboost.py`, `data/`.

---

### `common/change_rate_data.py` — Thay đổi tỷ lệ mất cân bằng
- **Chức năng**: Hàm `change_rate_data(X, y, new_rate)` giảm số mẫu lớp thiểu số (positive, y=1) để tạo tỷ lệ mất cân bằng mong muốn.
- **Cơ chế**: Shuffle ngẫu nhiên mẫu positive → giữ lại `N_neg * new_rate` mẫu → nối với tất cả mẫu negative.
- **Được gọi bởi**: Tất cả data loader.

---

### `__main__.py` — Script thí nghiệm chính
- **Chức năng**: Chạy so sánh nhiều biến thể AdaBoost:
  - AdaBoost + WSVM (bài báo 2016)
  - Nova 1 + WSVM (proposed preprocessing)
  - Nova 1+2 + WSVM (cả hai cải tiến)
- **Cấu hình**: M=10, C=10000, theta=0.2, duyệt nhiều imbalance rate.
- **Đầu ra**: File `.txt` chứa metric trung bình qua nhiều lần chạy.
- **Dataset**: Vertebral Column (có thể chuyển sang các dataset khác bằng comment/uncomment).

### `__main1__.py` — Script thí nghiệm (so sánh đầy đủ 8 biến thể)
- **Chức năng**: So sánh 8 biến thể: ADA+SVM, ADA+WSVM, Nova1+SVM, Nova1+WSVM, Nova2+SVM, Nova2+WSVM, Nova1+2+SVM, Nova1+2+WSVM.
- **Dataset**: Churn (có thể chuyển).

### `__main2__.py` — Script thí nghiệm (có confusion matrix)
- **Chức năng**: Tương tự `__main__.py` nhưng thêm tính confusion matrix cho mỗi biến thể.
- **Dataset**: Vertebral Column.

### `adaboosttree.py` — Baseline AdaBoost + Decision Tree
- **Chức năng**: Chạy AdaBoost chuẩn (`sklearn.AdaBoostClassifier`) làm so sánh baseline.
- **Dataset**: Co-Author (có thể chuyển).

### `adaboostWSVM.py` — AdaBoost + WSVM thí nghiệm
- **Chức năng**: Tương tự `__main__.py`, chạy trên Indian Liver Patient, so sánh WSVM và Nova1+SVM.

---

### Notebooks

| Notebook | Chức năng |
|----------|-----------|
| `FullCode_ImADA.ipynb` | Pipeline đầy đủ: load data → train Im.AdaBoost → evaluate. Notebook chính. |
| `FullCode_ImADA_ver1.ipynb` | Phiên bản 1 với cấu hình khác. |
| `FullCode_ImADA_ver2.ipynb` | Phiên bản 2 với cải tiến bổ sung. |
| `FullCode_ImADA_UCI.ipynb` | Thí nghiệm mở rộng trên nhiều tập dữ liệu UCI. |
| `Train_CNN.ipynb` | Huấn luyện mạng CNN cho bài toán phân loại không cân bằng. |

---

### Các Data Loader (`data/*.py`)

Mỗi file loader có cùng pattern:

```
load_data(test_size, new_rate)    # Phiên bản TestSize + IR
load_data(test_size)              # Phiên bản TestSize
load_data()                       # Phiên bản KFold (trả X, y toàn bộ)
```

| File | Dataset | Label mapping | Đặc biệt |
|------|---------|---------------|-----------|
| `Vertebral_column.py` | Vertebral Column | Abnormal→-1, Normal→1 | — |
| `Co_Author.py` | Co-Author | -1→1, 1→-1 (đảo) | StandardScaler |
| `Ecoli_TestSize.py` | E.coli | im→1, others→-1 | StandardScaler |
| `Haberman_TestSize.py` | Haberman Survival | 2→1, 1→-1 | StandardScaler |
| `churn.py` | Churn | True→1, False→-1 | LabelEncoder, PCA(15) |
| `indian_liver_patient.py` | Indian Liver Patient | 1→-1, 2→1 | Imputer, PCA(6) |
| `seismic_bumps.py` | Seismic Bumps | 1→1, 0→-1 | LabelEncoder, PCA(11) |
| `spect_heart.py` | SPECT Heart | 1→-1, 0→1 | — |
| `Pima.py` | Pima Diabetes | 1→1, 0→-1 | — |
| `Transfution_TestSize.py` | Blood Transfusion | 1→1, 0→-1 | StandardScaler |
| `Yeast_TestSize.py` | Yeast | ME2→1, others→-1 | StandardScaler |

---

## 6. Luồng thực thi của hệ thống

### Luồng chính (chạy từ script)

```
__main__.py (hoặc __main1__.py, __main2__.py)
 │
 ├── Import data loader (VD: Vertebral_column)
 ├── Cấu hình: M=10, C=10000, theta=0.2
 │
 ├── Vòng lặp theta
 │   ├── Vòng lặp imbalance rate (mr)
 │   │   ├── Vòng lặp test_size (i)
 │   │   │   ├── Vòng lặp kf (lặp nhiều lần = cross-validation thủ công)
 │   │   │   │   │
 │   │   │   │   ├── data_loader.load_data(test_size, new_rate)
 │   │   │   │   │   ├── Đọc CSV
 │   │   │   │   │   ├── Map labels → {-1, +1}
 │   │   │   │   │   ├── change_rate_data(X, y, new_rate)
 │   │   │   │   │   ├── train_test_split()
 │   │   │   │   │   └── StandardScaler / PCA (tùy dataset)
 │   │   │   │   │
 │   │   │   │   ├── toa.fit(X_train, y_train, M, C, ...)
 │   │   │   │   │   ├── methods.intinitialization_weight_adjustment()
 │   │   │   │   │   ├── [methods.intinitialization_instance_categorization()]
 │   │   │   │   │   ├── Loop M lần:
 │   │   │   │   │   │   ├── svm.fit(X, y, C, distribution_weight)
 │   │   │   │   │   │   ├── methods.predict_svm()
 │   │   │   │   │   │   ├── methods.find_true_false_index()
 │   │   │   │   │   │   ├── methods.confident()  → alpha_i
 │   │   │   │   │   │   ├── methods.update_weight_adjustment()
 │   │   │   │   │   │   └── [methods.update_instance_categorization_final()]
 │   │   │   │   │   └── Return (w, b, alpha)
 │   │   │   │   │
 │   │   │   │   ├── toa.predict(X_test, w, b, alpha, M)
 │   │   │   │   │   └── H(x) = sign(Σ alpha_i * (x·w_i + b_i))
 │   │   │   │   │
 │   │   │   │   └── precision_recall_fscore_support(y_test, pred)
 │   │   │   │
 │   │   │   └── Tính trung bình metric qua kf lần chạy
 │   │   │
 │   │   └── Ghi kết quả ra file .txt
```

### Luồng notebook

```
FullCode_ImADA.ipynb
 → Import thư viện
 → Load dataset
 → Chạy Im.AdaBoost với nhiều cấu hình
 → Vẽ biểu đồ so sánh
 → Phân tích kết quả
```

---

## 7. Phụ thuộc và thư viện sử dụng

### Python Packages

| Package | Phiên bản | Mục đích |
|---------|-----------|----------|
| `numpy` | — | Tính toán ma trận, vector |
| `pandas` | — | Đọc/xử lý dữ liệu CSV |
| `scikit-learn` | — | SVC, train_test_split, StandardScaler, PCA, LabelEncoder, metrics |
| `cvxopt` | — | Giải bài toán Quadratic Programming cho SVM |
| `keras` / `tensorflow` | — | CNN training, to_categorical |
| `matplotlib` | — | Vẽ biểu đồ (trong notebook) |

### Framework
- **scikit-learn**: Preprocessing, evaluation metrics, baseline classifiers
- **cvxopt**: SVM custom solver (QP)
- **Keras/TensorFlow**: CNN models

---

## 8. Hướng dẫn chạy dự án

### 1. Cài đặt môi trường

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
pip install numpy pandas scikit-learn cvxopt matplotlib
pip install tensorflow keras   # Nếu cần chạy CNN
```

### 3. Cấu hình đường dẫn dữ liệu

> **Quan trọng**: Các file data loader trong `data/` sử dụng đường dẫn tuyệt đối (ví dụ `D:/MULTIMEDIA/...`). Bạn cần:
> - Sửa đường dẫn trong các file `data/*.py` cho phù hợp với máy của bạn, hoặc
> - Đặt file CSV vào đúng vị trí được chỉ định.
>
> File CSV nằm trong `data/datasets/`.

### 4. Chạy thí nghiệm chính

```bash
# Chạy script so sánh các biến thể AdaBoost
python __main__.py

# Hoặc chạy các phiên bản khác
python __main1__.py
python __main2__.py

# Chạy AdaBoost + Decision Tree baseline
python adaboosttree.py

# Tìm C tối ưu cho SVM
python Find_C.py
```

### 5. Chạy notebook

```bash
jupyter notebook FullCode_ImADA.ipynb
```

### 6. Tùy chỉnh thí nghiệm

Trong các file `__main__*.py`, thay đổi:
- `M`: Số vòng lặp AdaBoost (mặc định: 10)
- `C`: Hệ số regularization SVM (mặc định: 10000)
- `theta`: Tham số điều chỉnh trọng số (mặc định: 0.2)
- `new_rate = 1/mr`: Tỷ lệ mất cân bằng (VD: 1/9 = ~11% minority)
- Comment/uncomment dòng load data để chuyển dataset

### 7. Đọc kết quả

- File `.txt` được tạo tại thư mục gốc với tên dạng: `{Dataset}_theta_{theta}_rate_1_{mr}.txt`
- Kết quả CSV trong `Experiment/`

---

## Ghi chú

- Dự án phục vụ **nghiên cứu học thuật**, code có tính chất thử nghiệm.
- Các đường dẫn file trong data loader cần được cập nhật cho phù hợp với hệ thống của bạn.
- Label convention: **+1 = minority (positive)**, **-1 = majority (negative)**.
- Tham số `theta` kiểm soát mức độ ưu tiên cho lớp thiểu số trong khởi tạo trọng số.
- Tham số `proposed_alpha` thay đổi cách tính confident, tăng penalty khi phân loại sai lớp thiểu số.
