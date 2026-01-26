# %% [markdown]
# # Bước 2: Nhập và Kiểm Tra Dữ Liệu
# 
# Bước này tập trung vào việc tải dữ liệu và kiểm tra ban đầu.
# 
# **Mục tiêu:**
# - Tải dữ liệu không bị lỗi
# - Kiểm tra kích thước (rows, columns)
# - Xác định kiểu dữ liệu
# - Phát hiện lỗi hoặc không nhất quán

# %% [markdown]
# ## 2.1 Import thư viện

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

print("✅ Thư viện đã được import thành công!")

# %% [markdown]
# ## 2.2 Tải dữ liệu

# %%
DATA_PATH = Path("../data/raw/accepted_2007_to_2018Q4.csv")

print("=" * 70)
print("📂 THÔNG TIN FILE")
print("=" * 70)
print(f"Đường dẫn: {DATA_PATH}")
print(f"File tồn tại: {DATA_PATH.exists()}")

# Đọc sample để phân tích (file gốc ~1.6GB)
SAMPLE_SIZE = 50000
df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE, low_memory=False)

print(f"✅ Đã tải thành công {SAMPLE_SIZE:,} dòng!")

# %% [markdown]
# ## 2.3 Kiểm tra kích thước dữ liệu

# %%
print("=" * 70)
print("📐 KÍCH THƯỚC DỮ LIỆU")
print("=" * 70)
print(f"Số hàng: {df.shape[0]:,}")
print(f"Số cột: {df.shape[1]}")
print(f"Tổng số ô (cells): {df.shape[0] * df.shape[1]:,}")
print(f"Bộ nhớ sử dụng: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# %% [markdown]
# ## 2.4 Xem dữ liệu mẫu

# %%
print("=" * 70)
print("👀 5 DÒNG ĐẦU TIÊN")
print("=" * 70)
display(df.head())

# %%
print("\n" + "=" * 70)
print("👀 5 DÒNG CUỐI CÙNG")
print("=" * 70)
display(df.tail())

# %% [markdown]
# ## 2.5 Thông tin chi tiết từng cột

# %%
print("=" * 70)
print("ℹ️ THÔNG TIN CHI TIẾT (df.info())")
print("=" * 70)
df.info()

# %% [markdown]
# ## 2.6 Phân loại kiểu dữ liệu

# %%
print("=" * 70)
print("🔤 PHÂN LOẠI KIỂU DỮ LIỆU")
print("=" * 70)

dtype_summary = df.dtypes.value_counts()
print("\nSố cột theo kiểu dữ liệu:")
for dtype, count in dtype_summary.items():
    print(f"  {dtype}: {count} cột")

# %%
# Liệt kê chi tiết
print("\n--- Cột số (Numerical) ---")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Tổng: {len(num_cols)} cột")
for col in num_cols:
    print(f"  • {col}: {df[col].dtype}")

# %%
print("\n--- Cột phân loại (Object) ---")
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Tổng: {len(cat_cols)} cột")
for col in cat_cols:
    nunique = df[col].nunique()
    print(f"  • {col}: {nunique} giá trị duy nhất")

# %% [markdown]
# ## 2.7 Số lượng giá trị duy nhất

# %%
print("=" * 70)
print("🔢 SỐ LƯỢNG GIÁ TRỊ DUY NHẤT (df.nunique())")
print("=" * 70)

unique_counts = df.nunique().sort_values()
print(unique_counts)

# %% [markdown]
# ## 2.8 Thống kê mô tả cơ bản

# %%
print("=" * 70)
print("📊 THỐNG KÊ MÔ TẢ - BIẾN SỐ (df.describe())")
print("=" * 70)
display(df.describe())

# %%
print("\n" + "=" * 70)
print("📊 THỐNG KÊ MÔ TẢ - TẤT CẢ BIẾN")
print("=" * 70)
display(df.describe(include='all'))

# %% [markdown]
# ## 2.9 Phát hiện lỗi và không nhất quán

# %%
print("=" * 70)
print("⚠️ PHÁT HIỆN LỖI VÀ KHÔNG NHẤT QUÁN")
print("=" * 70)

# Kiểm tra giá trị âm ở các cột không nên âm
potential_positive_cols = ['loan_amnt', 'funded_amnt', 'annual_inc', 'installment']
print("\n1️⃣ Kiểm tra giá trị âm không hợp lệ:")
for col in potential_positive_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        print(f"   {col}: {negative_count} giá trị âm")

# Kiểm tra giá trị ngoài phạm vi cho int_rate
if 'int_rate' in df.columns:
    print("\n2️⃣ Kiểm tra lãi suất (int_rate):")
    print(f"   Min: {df['int_rate'].min()}, Max: {df['int_rate'].max()}")
    out_of_range = ((df['int_rate'] < 0) | (df['int_rate'] > 100)).sum()
    print(f"   Giá trị ngoài [0, 100]: {out_of_range}")

# Kiểm tra term
if 'term' in df.columns:
    print("\n3️⃣ Kiểm tra kỳ hạn (term):")
    print(df['term'].value_counts())

# %% [markdown]
# ## 2.10 Tổng kết Bước 2

# %%
print("=" * 70)
print("📋 TỔNG KẾT BƯỚC 2: NHẬP VÀ KIỂM TRA DỮ LIỆU")
print("=" * 70)
print(f"""
✅ ĐÃ HOÀN THÀNH:
   • Tải thành công {SAMPLE_SIZE:,} dòng từ dataset
   • Kích thước: {df.shape[0]:,} hàng × {df.shape[1]} cột
   • Phân loại: {len(num_cols)} biến số, {len(cat_cols)} biến phân loại
   • Bộ nhớ: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

📊 PHÁT HIỆN BAN ĐẦU:
   • Nhiều cột có giá trị thiếu (sẽ xử lý ở Bước 3)
   • Một số cột date đang ở dạng object
   • Cần kiểm tra thêm về outliers

📝 BƯỚC TIẾP THEO:
   → Bước 3: Xử lý giá trị thiếu (Missing Values)
""")
