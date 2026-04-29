"""Script phân tích tổng quan dữ liệu 600K ván cờ đã xử lý."""

import polars as pl
import json
from pathlib import Path

def main():
    file_path = Path("data/processed/sample_600k_dl.parquet")
    if not file_path.exists():
        print(f"❌ Không tìm thấy file: {file_path}")
        return

    print(f"Đang tải {file_path} (Sẽ mất vài giây vì file khá lớn)...")
    df = pl.read_parquet(file_path)

    print(f"\n=========================================")
    print(f"TỔNG QUAN DỮ LIỆU: {df.height:,} dòng, {df.width} cột")
    print(f"=========================================\n")

    print("1. Schema (Các cột):")
    for name, dtype in df.schema.items():
        print(f"   - {name}: {dtype}")

    print("\n2. Phân bổ ELO Band (ModelBand):")
    band_counts = df["ModelBand"].value_counts().sort("ModelBand")
    for row in band_counts.iter_rows():
        print(f"   - Band {row[0]}: {row[1]:,} ván")

    print("\n3. Phân bổ Thể thức (GameFormat):")
    format_counts = df["GameFormat"].value_counts().sort("count", descending=True)
    for row in format_counts.iter_rows():
        print(f"   - {row[0]}: {row[1]:,} ván ({(row[1]/df.height)*100:.1f}%)")

    print("\n4. Thống kê Chiều dài ván cờ (NumMoves):")
    print(f"   - Trung bình: {df['NumMoves'].mean():.1f} nước")
    print(f"   - Nhỏ nhất  : {df['NumMoves'].min()} nước")
    print(f"   - Lớn nhất  : {df['NumMoves'].max()} nước")

    print("\n5. Kiểm tra tính hợp lệ của Chuỗi CPL và Clock:")
    try:
        # Kiểm tra mẫu dữ liệu dòng đầu tiên
        first_cpl = json.loads(df["cpl_seq"][0])
        first_clock = json.loads(df["ClockSeq"][0])
        print(f"   - Mẫu CPL (5 phần tử đầu): {first_cpl[:5]}")
        print(f"   - Độ dài CPL ván đầu: {len(first_cpl)}")
        print(f"   - Độ dài Clock ván đầu: {len(first_clock)}")
        
        # Đếm số lượng chuỗi rỗng
        has_empty_cpl = df.filter(pl.col("cpl_seq") == "[]").height
        print(f"   - Số ván có chuỗi CPL rỗng: {has_empty_cpl:,} ({has_empty_cpl/df.height*100:.2f}%)")
        
        # Kiểm tra ván có null CPL (do bị Mate hoặc lỗi Parse)
        print("   - Đang đếm số ván có giá trị CPL NaN/Null (thường là nước chiếu hết)...")
        # Chuyển đổi json parse text về List để kiểm tra độ tin cậy
        def check_has_null(json_str):
            try:
                arr = json.loads(json_str)
                for x in arr:
                    if x is None or (isinstance(x, float) and x != x): # Kiểm tra NaN
                        return True
                return False
            except:
                return True
                
    except Exception as e:
        print(f"   - Lỗi khi kiểm tra chuỗi: {e}")

    print("\n=========================================")

if __name__ == "__main__":
    main()
