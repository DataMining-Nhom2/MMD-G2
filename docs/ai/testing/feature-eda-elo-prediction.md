---
phase: testing
title: Testing Strategy — EDA cho dự đoán ELO realtime
description: Chiến lược kiểm tra chất lượng và tính chính xác của EDA notebook
---

# Testing Strategy — EDA cho Dự đoán ELO Realtime

## Test Coverage Goals
**Mức độ kiểm tra hướng tới**

- **Data Loading**: Verify data loads correctly, schema matches expected, no silent corruption
- **Sampling Quality**: Stratified sample đại diện cho toàn bộ dataset (chi-square test phân phối)
- **Visualization**: Mỗi biểu đồ render đúng, có title/label/legend, export thành công
- **Insight Quality**: Mỗi insight có factual basis, không mâu thuẫn nội bộ
- **Reproducibility**: Cùng seed → cùng kết quả mỗi lần chạy
- **Performance**: Notebook chạy end-to-end < 10 phút (CPU 20 threads + GPU RTX 3060)

## Unit Tests
**Kiểm tra từng thành phần**

### Data Loading & Quality
- [x] Verify 2 file Parquet load thành công (không FileNotFoundError)
- [x] Verify schema 14 cột đúng tên và kiểu dữ liệu
- [x] Verify total rows ~187M (±1% do data updates)
- [x] Verify null count cho từng cột — WhiteElo, BlackElo phải = 0 (đã filter ở preprocessing)
- [x] Verify ELO range hợp lý (400 ≤ ELO ≤ 3500)

### Sampling
- [x] Verify sample size = SAMPLE_SIZE (hoặc gần đúng nếu stratified)
- [x] Verify mỗi ELO band có ≥ 100 samples (không bị bỏ sót ELO extreme)
- [x] Verify mỗi GameFormat có representation trong sample
- [x] Verify reproducibility: chạy 2 lần cùng seed → cùng sample

### Derived Features
- [x] EloDiff = WhiteElo - BlackElo (kiểm tra tính đúng trên 100 rows)
- [x] EloBand mapping đúng: EloAvg=1500 → band "1400-1600"
- [x] EcoCategory: ECO="B20" → EcoCategory="B"
- [x] extract_first_n_moves("1. e4 e5 2. Nf3 Nc6", 4) == "e4 e5 Nf3 Nc6"
- [x] extract_first_n_moves("", 10) == "" (edge case: ván trống)

### Move Parsing
- [x] extract_first_n_moves xử lý đúng SAN format chuẩn
- [x] Xử lý game endings (1-0, 0-1, 1/2-1/2) — không include vào moves
- [x] Xử lý castling notation (O-O, O-O-O)
- [x] Xử lý promotion (e8=Q)
- [x] Performance: parse 100K moves < 30s

## Integration Tests
**Kiểm tra tương tác giữa các components**

- [x] Data load → Sample → Derived columns → Plot pipeline chạy end-to-end
- [x] LazyFrame aggregation trên full dataset trả về kết quả consistent với sample
- [x] Biểu đồ export ra eda/outputs/ thành công (file exists, size > 0)
- [x] Notebook restart kernel & run all → tất cả cells pass

## End-to-End Tests
**Kiểm tra toàn bộ flow**

- [x] **Full Notebook Run**: `Run All Cells` thành công không lỗi
- [x] **Output Validation**: ≥ 8 file PNG/SVG trong eda/outputs/
- [x] **Insight Coverage**: Mỗi biểu đồ có Markdown insight cell ngay sau
- [x] **3 Strategic Questions**: Q1, Q2, Q3 được trả lời rõ ràng trong Executive Summary
- [x] **Feature Ranking**: Bảng ranking feature candidates xuất hiện trong notebook

## Test Data
**Dữ liệu dùng cho testing**

### Main Data
- `data/processed/lichess_2025-12_ml.parquet` (93.9M rows)
- `data/processed/lichess_2026-01_ml.parquet` (93.4M rows)

### Validation Checks
```python
# Quick validation script — chạy trước notebook
import polars as pl
from src.config import DATA_PROCESSED

lf = pl.scan_parquet(DATA_PROCESSED / "*.parquet")
schema = lf.collect_schema()

# Schema check
assert len(schema) == 14, f"Expected 14 cols, got {len(schema)}"
assert "WhiteElo" in schema, "Missing WhiteElo"
assert "Moves" in schema, "Missing Moves"

# Row count check
total = lf.select(pl.len()).collect().item()
assert total > 100_000_000, f"Too few rows: {total}"

# ELO range check
elo_stats = lf.select(
    pl.col("WhiteElo").min().alias("min_elo"),
    pl.col("WhiteElo").max().alias("max_elo"),
).collect()
assert elo_stats["min_elo"][0] >= 400, "ELO too low"
assert elo_stats["max_elo"][0] <= 3500, "ELO too high"

print("All validation checks passed!")
```

## Test Reporting & Coverage
**Báo cáo kiểm tra**

### Checklist sau khi hoàn thành
- [x] Notebook chạy "Restart Kernel & Run All" thành công
- [x] Tất cả biểu đồ hiển thị đúng (không blank, không error)
- [x] Tất cả Markdown cells format đúng
- [x] Biểu đồ export ra eda/outputs/ đầy đủ
- [x] Không có FutureWarning hoặc DeprecationWarning
- [x] RAM usage peak < 20 GB (kiểm tra bằng resource monitor)
- [x] VRAM usage peak < 10 GB (kiểm tra bằng nvidia-smi)

### Coverage Gaps (ghi nhận)
- Move parsing chỉ test trên sample nhỏ — cần verify trên full dataset ở Phase 4
- Feature importance dùng XGBoost GPU (hoặc fallback DecisionTree) — chỉ là proxy, không phải ground truth
- GPU fallback path cần test riêng: disable CUDA → verify DecisionTree CPU tự động kích hoạt

## Manual Testing
**Kiểm tra bằng mắt**

### Visual Quality Checklist
- [x] Mỗi biểu đồ có Title rõ ràng
- [x] Mỗi biểu đồ có X-axis label và Y-axis label
- [x] Mỗi biểu đồ có Legend (nếu nhiều series)
- [x] Font size đủ lớn để đọc
- [x] Màu sắc phân biệt rõ (không bị trùng)
- [x] Annotation cho data points quan trọng
- [x] Không bị cắt text (bbox_inches='tight')

### Insight Quality Checklist
- [x] Mỗi insight bắt đầu bằng quan sát factual
- [x] Mỗi insight có giải thích ý nghĩa cờ vua
- [x] Mỗi insight có hàm ý cho mô hình ML
- [x] Không có mâu thuẫn giữa các insights
- [x] Executive Summary tổng hợp coherent

## Performance Testing
**Kiểm tra hiệu năng**

| Metric | Target | Cách đo |
|--------|--------|---------|
| Notebook total runtime | < 10 phút | %%time trên cell đầu/cuối |
| Data loading time | < 30s cho sample | %%time trên cell load |
| Peak RAM usage | < 20 GB | htop / resource monitor |
| Peak VRAM usage | < 10 GB | nvidia-smi |
| Biểu đồ render time | < 10s mỗi biểu đồ | %%time trên cell plot |
| Move parsing (100K) | < 30s | %%time trên cell parse |
| XGBoost GPU training (1M rows) | < 60s | %%time trên cell model |

## Bug Tracking
**Quản lý vấn đề**

### Severity Levels
- **P0 (Critical)**: Notebook crash, data corruption, OOM → Fix ngay
- **P1 (High)**: Biểu đồ sai dữ liệu, insight mâu thuẫn → Fix trước submit
- **P2 (Medium)**: Warning messages, slow performance → Fix nếu có thời gian
- **P3 (Low)**: Cosmetic issues, typo → Fix later

### Regression Testing
- Sau mỗi thay đổi code: "Restart Kernel & Run All"
- Verify output consistency (biểu đồ + insights không thay đổi ngoài ý muốn)
