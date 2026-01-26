# %% [markdown]
# # Bước 1: Hiểu Rõ Vấn Đề và Dữ Liệu
# 
# Bước đầu tiên trong EDA là hiểu rõ mục tiêu phân tích và dữ liệu có sẵn.
# 
# **Mục tiêu:**
# - Xác định câu hỏi nghiên cứu/kinh doanh
# - Hiểu ý nghĩa các biến
# - Nhận diện vấn đề chất lượng dữ liệu

# %% [markdown]
# ## 1.1 Thư viện cần thiết

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
print("✅ Thư viện đã được import thành công!")

# %% [markdown]
# ## 1.2 Xác định vấn đề nghiên cứu

# %%
# =====================================================================
# CÂU HỎI NGHIÊN CỨU / MỤC TIÊU KINH DOANH
# =====================================================================
print("""
📌 MỤC TIÊU PHÂN TÍCH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset: Lending Club Loan Data (2007-2018 Q4)

🎯 Câu hỏi nghiên cứu chính:
   1. Những yếu tố nào ảnh hưởng đến khả năng trả nợ của khách hàng?
   2. Có thể dự đoán khoản vay nào sẽ bị default không?
   3. Mức lãi suất được xác định như thế nào?

📊 Ứng dụng thực tế:
   - Đánh giá rủi ro tín dụng
   - Phân loại khách hàng
   - Tối ưu hóa chính sách cho vay
""")

# %% [markdown]
# ## 1.3 Tải và xem thông tin ban đầu

# %%
# Đường dẫn dữ liệu
DATA_PATH = Path("../data/raw/accepted_2007_to_2018Q4.csv")

# Đọc một số dòng đầu để hiểu cấu trúc
df_head = pd.read_csv(DATA_PATH, nrows=5)

print("=" * 70)
print("📋 CÁC CỘT TRONG DATASET")
print("=" * 70)
for i, col in enumerate(df_head.columns, 1):
    print(f"{i:3d}. {col}")

print(f"\n📊 Tổng số cột: {len(df_head.columns)}")

# %% [markdown]
# ## 1.4 Ý nghĩa các biến chính

# %%
# =====================================================================
# TỪ ĐIỂN DỮ LIỆU (DATA DICTIONARY)
# =====================================================================
print("""
📚 Ý NGHĨA CÁC BIẾN QUAN TRỌNG:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 THÔNG TIN KHOẢN VAY:
   • loan_amnt       : Số tiền vay
   • funded_amnt     : Số tiền được cấp
   • term            : Kỳ hạn vay (36 hoặc 60 tháng)
   • int_rate        : Lãi suất (%)
   • installment     : Số tiền trả góp hàng tháng
   • loan_status     : Trạng thái khoản vay (TARGET VARIABLE)

👤 THÔNG TIN NGƯỜI VAY:
   • annual_inc      : Thu nhập hàng năm
   • emp_length      : Thời gian làm việc
   • home_ownership  : Tình trạng nhà ở
   • verification_status : Trạng thái xác minh thu nhập

📊 ĐIỂM TÍN DỤNG:
   • grade           : Hạng tín dụng (A-G)
   • sub_grade       : Hạng tín dụng chi tiết (A1-G5)
   • dti             : Tỷ lệ nợ trên thu nhập
   • revol_util      : Tỷ lệ sử dụng tín dụng quay vòng

🎯 BIẾN MỤC TIÊU:
   • loan_status: Trạng thái cuối cùng của khoản vay
     - Fully Paid: Đã trả hết ✓
     - Charged Off: Xóa nợ (default) ✗
     - Current: Đang trong kỳ hạn
     - Late: Trễ hạn
""")

# %% [markdown]
# ## 1.5 Xác định loại dữ liệu

# %%
# Đọc mẫu lớn hơn để phân tích
df_sample = pd.read_csv(DATA_PATH, nrows=1000, low_memory=False)

print("=" * 70)
print("🔤 PHÂN LOẠI BIẾN THEO KIỂU DỮ LIỆU")
print("=" * 70)

# Phân loại biến
numerical = df_sample.select_dtypes(include=[np.number]).columns.tolist()
categorical = df_sample.select_dtypes(include=['object']).columns.tolist()

print(f"\n📊 Biến số (Numerical): {len(numerical)} cột")
print(f"📝 Biến phân loại (Categorical): {len(categorical)} cột")

# %%
# Chi tiết từng loại
print("\n--- Biến số ---")
for col in numerical[:10]:
    print(f"  • {col}")
if len(numerical) > 10:
    print(f"  ... và {len(numerical) - 10} biến khác")

print("\n--- Biến phân loại ---")
for col in categorical[:10]:
    print(f"  • {col}")
if len(categorical) > 10:
    print(f"  ... và {len(categorical) - 10} biến khác")

# %% [markdown]
# ## 1.6 Xác định vấn đề chất lượng dữ liệu ban đầu

# %%
print("=" * 70)
print("⚠️ VẤN ĐỀ CHẤT LƯỢNG DỮ LIỆU CẦN CHÚ Ý")
print("=" * 70)

# Kiểm tra nhanh missing values
missing_pct = (df_sample.isnull().sum() / len(df_sample) * 100).sort_values(ascending=False)
high_missing = missing_pct[missing_pct > 50]

print(f"""
📋 TỔNG QUAN VẤN ĐỀ:

1️⃣ Giá trị thiếu (Missing Values):
   • Số cột có >50% missing: {len(high_missing)}
   • Cần xử lý trong Bước 3

2️⃣ Kiểu dữ liệu:
   • Nhiều cột date đang ở dạng object
   • Cần chuyển đổi trong Bước 4

3️⃣ Ngoại lệ (Outliers):
   • annual_inc có thể có giá trị cực đoan
   • Cần kiểm tra trong Bước 7

4️⃣ Mất cân bằng (Imbalance):
   • loan_status có thể không cân bằng
   • Cần xem xét khi modeling
""")

# %% [markdown]
# ## 1.7 Tổng kết Bước 1

# %%
print("=" * 70)
print("📋 TỔNG KẾT BƯỚC 1: HIỂU RÕ VẤN ĐỀ VÀ DỮ LIỆU")
print("=" * 70)
print(f"""
✅ ĐÃ HOÀN THÀNH:
   • Xác định mục tiêu phân tích: Dự đoán rủi ro tín dụng
   • Hiểu ý nghĩa {len(df_sample.columns)} biến trong dataset
   • Phân loại: {len(numerical)} biến số, {len(categorical)} biến phân loại
   • Nhận diện vấn đề chất lượng dữ liệu ban đầu

🎯 BIẾN MỤC TIÊU: loan_status
   • Binary classification: Default vs Non-default

📝 CÁC BƯỚC TIẾP THEO:
   → Bước 2: Nhập và kiểm tra dữ liệu chi tiết
   → Bước 3: Xử lý giá trị thiếu
""")
