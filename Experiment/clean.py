import os
import pandas as pd

input_folder = r"D:\research\Thu-Nghiem\Experiment"
output_folder = r"D:\research\Thu-Nghiem\Experiment\output"

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(input_folder)

# Lọc file CSV
csv_files = [f for f in files if f.endswith(".csv")]

print("📄 CSV files:", csv_files)

if not csv_files:
    print("❌ Không tìm thấy file CSV")
    exit()

for file in csv_files:
    input_path = os.path.join(input_folder, file)
    print(f"\n➡️ Đang xử lý: {file}")

    try:
        # Đọc CSV
        try:
            df = pd.read_csv(input_path, encoding="utf-8-sig")
        except:
            df = pd.read_csv(input_path, encoding="latin1")

        # Clean column
        df.columns = df.columns.str.strip()

        print("Columns:", df.columns.tolist())

        # Kiểm tra cột
        if "Method" not in df.columns or "SE" not in df.columns:
            print("⚠️ Thiếu Method hoặc SE → bỏ qua")
            continue

        # Convert SE về số (tránh lỗi sort)
        df["SE"] = pd.to_numeric(df["SE"], errors="coerce")

        # Sort theo Method + SE giảm dần
        sorted_df = df.sort_values(
            by=["Method", "SE"],
            ascending=[True, False]
        )

        # Tạo file output
        output_name = os.path.splitext(file)[0] + "_sorted.csv"
        output_path = os.path.join(output_folder, output_name)

        # Xuất CSV
        sorted_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"✅ Đã lưu: {output_path}")

    except Exception as e:
        print(f"❌ Lỗi file {file}: {e}")