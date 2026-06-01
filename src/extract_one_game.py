import polars as pl
from pathlib import Path
import sys

# Đường dẫn mặc định (Sửa lại nếu file Parquet nằm ở ổ D, E...)
PARQUET_FILE = Path("data/processed/sample_30k.parquet")

def main():
    if not PARQUET_FILE.exists():
        print(f"❌ KHÔNG TÌM THẤY FILE: {PARQUET_FILE}")
        print("💡 Gợi ý: Hãy sửa biến PARQUET_FILE trong script này trỏ đúng vào file thật trên máy bạn.")
        sys.exit(1)

    print(f"Đang đọc file: {PARQUET_FILE}...")
    
    # Dùng Lazy Scan để chỉ bốc đúng 1 dòng (dòng 500 cho nó random), cực kỳ nhẹ RAM
    try:
        df_sample = pl.scan_parquet(PARQUET_FILE).slice(500, 1).collect()
        
        # Nếu thư mục 500 không có, lấy dòng 0
        if df_sample.height == 0:
            df_sample = pl.scan_parquet(PARQUET_FILE).head(1).collect()
            
    except Exception as e:
        print(f"❌ Lỗi khi đọc file Parquet: {e}")
        sys.exit(1)

    if df_sample.height == 0:
        print("❌ File Parquet trống không!")
        sys.exit(1)

    row = df_sample.row(0, named=True)
    
    print("\n" + "="*70)
    print(" 🔎 BÁO CÁO SOI 1 VÁN CỜ TRONG PARQUET ".center(70))
    print("="*70 + "\n")
    
    for key, value in row.items():
        if key == "Moves":
            continue # In sau cùng cho dễ nhìn
        print(f"  ▪️ {key:<15}: {value}")
        
    print(f"\n  ▪️ Moves:")
    
    moves_str = str(row.get("Moves", ""))
    print(f"    {moves_str[:800]} ... [Cắt bớt nếu quá dài]\n")
    
    print("="*70)
    print(" 🛠 KIỂM TRA SỨC KHỎE DATA (CLOCK & EVAL) ".center(70))
    print("="*70)
    
    has_clk = "[%clk" in moves_str
    has_eval = "[%eval" in moves_str
    
    print(f" \n  [Đồng hồ suy nghĩ]  Chứa tag [%clk] ?  --->  {'🟢 CÓ MẶT (Quá ngon!)' if has_clk else '🔴 MẤT TÍCH (Nguy hiểm!)'}")
    print(f"  [Điểm Stockfish]    Chứa tag [%eval] ? --->  {'🟢 CÓ MẶT (Bắt được vàng!)' if has_eval else '🟡 KHÔNG CÓ (Trận này người dùng ko phân tích)'}\n")

if __name__ == "__main__":
    main()
