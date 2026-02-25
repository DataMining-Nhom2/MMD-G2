---
phase: implementation
title: Implementation Guide — EDA cho dự đoán ELO realtime
description: Hướng dẫn kỹ thuật chi tiết để implement notebook EDA
---

# Implementation Guide — EDA cho Dự đoán ELO Realtime

## Development Setup
**Cách bắt đầu**

### Prerequisites
```bash
# Activate conda environment
conda activate MMDS

# Kiểm tra libraries cơ bản
python -c "import polars, matplotlib, seaborn; print('OK')"

# Kiểm tra python-chess (cần cho Phase 4)
python -c "import chess; print('chess OK')" || pip install python-chess

# Kiểm tra sklearn (cần cho Phase 5 - fallback feature importance)
python -c "import sklearn; print('sklearn OK')" || pip install scikit-learn

# Kiểm tra GPU + XGBoost (cần cho Phase 5 - GPU feature importance)
nvidia-smi  # Expect: RTX 3060, 12GB VRAM
python -c "import xgboost as xgb; print('XGBoost', xgb.__version__)" || pip install xgboost

# Verify XGBoost GPU support
python -c "
import xgboost as xgb
import numpy as np
X = np.random.randn(100, 3)
y = np.random.randint(0, 3, 100)
model = xgb.XGBClassifier(device='cuda', n_estimators=10)
model.fit(X, y)
print('XGBoost GPU: OK')
"
```

### Cấu trúc files
```
eda/
├── EDA_Notebook.ipynb       # Notebook chính (tạo mới)
├── outputs/                 # Biểu đồ export (auto-created by config.py)
│   ├── elo_distribution.png
│   ├── eco_by_elo_band.png
│   ├── correlation_matrix.png
│   └── ...
data/processed/
├── lichess_2025-12_ml.parquet   # Input data (22.51 GB, 93.9M rows)
└── lichess_2026-01_ml.parquet   # Input data (22.42 GB, 93.4M rows)
```

## Code Structure
**Cách tổ chức code trong notebook**

### Cell Organisation Pattern
```
[Cell 1]  Markdown: Title & TOC
[Cell 2]  Code: Imports & Config
[Cell 3]  Code: Data Loading (Polars LazyFrame)
[Cell 4]  Code: Stratified Sampling
[Cell 5]  Code: Data Quality Check
[Cell 6]  Markdown: --- Section 2: Univariate ---
[Cell 7]  Code: ELO Distribution plot
[Cell 8]  Markdown: Insight cho ELO Distribution
[Cell 9]  Code: ECO Distribution plot
[Cell 10] Markdown: Insight cho ECO Distribution
...pattern lặp lại...
[Cell N-1] Markdown: Executive Summary
[Cell N]   Code: Export charts
```

### Naming Conventions
```python
# DataFrame naming
df_sample    # Stratified sample (2-5M rows)
df_full_agg  # Aggregated stats from full dataset (lazy)

# Plot naming
fig, ax = plt.subplots()  # Mỗi biểu đồ 1 figure
fig.savefig(EDA_OUTPUTS / "01_elo_distribution.png", dpi=150, bbox_inches='tight')

# Constants
SAMPLE_SIZE = 3_000_000
RANDOM_SEED = 42
ELO_BINS = [0, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 3500]
ELO_LABELS = ['<800', '800-1000', '1000-1200', '1200-1400', '1400-1600',
              '1600-1800', '1800-2000', '2000-2200', '2200-2500', '2500+']
```

## Implementation Notes
**Chi tiết kỹ thuật quan trọng**

### Data Loading — Tránh OOM
```python
import polars as pl
from src.config import DATA_PROCESSED, EDA_OUTPUTS

# Lazy scan — KHÔNG load vào RAM
lf = pl.scan_parquet(DATA_PROCESSED / "*.parquet")

# Aggregate statistics trên TOÀN BỘ dataset (lazy, chạy trên disk)
full_stats = lf.select([
    pl.col("WhiteElo").mean().alias("mean_white_elo"),
    pl.col("BlackElo").mean().alias("mean_black_elo"),
    pl.col("EloAvg").describe(),
    pl.len().alias("total_rows"),
]).collect()

# Stratified sample — CÓ load vào RAM nhưng chỉ 3M rows
df_sample = (
    lf
    .with_columns(
        pl.col("EloAvg")
        .cut(ELO_BINS, labels=ELO_LABELS)
        .alias("EloBand")
    )
    .collect()  # Phải collect trước khi sample
    .sample(n=SAMPLE_SIZE, seed=RANDOM_SEED)  # hoặc dùng fraction
)
```

> **Cảnh báo**: `.collect()` toàn bộ 187M rows sẽ cần ~30-40GB RAM. Nên dùng `.head(n)` hoặc filter trước khi collect. Hoặc dùng `streaming=True` nếu Polars version hỗ trợ.

### Sampling Strategy — Chi tiết
```python
# Phương án 1: Random sample (đơn giản nhưng có thể thiếu ELO cực)
df_sample = lf.head(5_000_000).collect()

# Phương án 2: Stratified (đảm bảo đại diện mọi ELO band)
# Bước 1: Scan chỉ metadata (không load Moves)
lf_meta = lf.select(pl.exclude("Moves"))

# Bước 2: Sample theo ELO band
# Cần collect rồi group_by sample — Polars chưa hỗ trợ stratified sample native
df_meta = lf_meta.collect(streaming=True)
df_sample = df_meta.group_by("GameFormat").map_groups(
    lambda g: g.sample(n=min(len(g), 500_000), seed=RANDOM_SEED)
)
```

### Visualization Template
```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Chuẩn hoá style cho toàn notebook
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

def save_plot(fig, name: str):
    """Lưu biểu đồ ra eda/outputs/ với format chuẩn."""
    path = EDA_OUTPUTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Đã lưu: {path}")
```

### Move Parsing — Performance Critical
```python
def extract_first_n_moves(moves_san: str, n: int = 10) -> str:
    """
    Trích xuất N nước đi đầu tiên từ SAN string.
    
    SAN format: "1. e4 e5 2. Nf3 Nc6 3. Bb5 ..."
    Mỗi "nước đi" = 1 ply (half-move). n=10 = 5 nước trắng + 5 nước đen.
    
    Dùng string split thay vì python-chess để performance tốt hơn 100×.
    """
    if not moves_san:
        return ""
    # Bỏ move numbers (1. 2. 3. ...) 
    tokens = moves_san.split()
    # Filter ra chỉ moves (không phải số thứ tự)
    moves_only = [t for t in tokens if not t[0].isdigit() and t not in ('1-0', '0-1', '1/2-1/2', '*')]
    return " ".join(moves_only[:n])

# Áp dụng trên Polars column (vectorized qua map_elements)
df_sample = df_sample.with_columns(
    pl.col("Moves")
    .map_elements(lambda m: extract_first_n_moves(m, 10), return_dtype=pl.Utf8)
    .alias("First10Moves")
)
```

### Feature Importance — GPU-Accelerated Assessment
```python
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Chuẩn bị features
features = ['WhiteElo', 'NumMoves', 'BaseTime', 'Increment']
X = df_sample.select(features).to_pandas()
y = df_sample['EloBand'].to_pandas()

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# XGBoost GPU — RTX 3060 12GB
# Tận dụng GPU cho training nhanh trên 1-3M rows
try:
    model = xgb.XGBClassifier(
        device='cuda',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y_encoded)
    engine = 'XGBoost GPU'
except Exception:
    # Fallback: DecisionTree CPU
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y_encoded)
    engine = 'DecisionTree CPU'

# Feature importance
importance = dict(zip(features, model.feature_importances_))
print(f"Feature Importance ({engine}):")
for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"  {feat:<20} {imp:.4f}")
```

## Patterns & Best Practices

### Pattern 1: Insight Markdown Template
```markdown
### 📊 Insight: [Tên biểu đồ]

**Quan sát**: [Mô tả factual]
**Ý nghĩa cờ vua**: [Giải thích nghiệp vụ]
**Hàm ý cho mô hình**: [Feature engineering / modeling implication]
**Action item**: [Bước tiếp theo cụ thể]
```

### Pattern 2: Lazy Aggregation cho Full Dataset
```python
# Tính value counts trên TOÀN BỘ 187M rows MÀ KHÔNG load hết vào RAM
eco_counts = (
    lf
    .group_by("ECO")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .head(20)
    .collect()  # Chỉ collect 20 rows
)
```

### Pattern 3: EDA Check Questions
Mỗi biểu đồ phải trả lời ít nhất 1 trong 3 câu hỏi:
1. **Feasibility**: Tín hiệu này có giúp predict ELO không?
2. **Imbalance**: Có bias/skew nào cần xử lý không?
3. **Feature Engineering**: Cần transform/create feature gì từ đây?

## Error Handling
**Xử lý lỗi thường gặp**

```python
# Parquet file không tồn tại
import os
for f in DATA_PROCESSED.glob("*.parquet"):
    print(f"  {f.name}: {os.path.getsize(f)/(1024**3):.2f} GB")

# Null values trong ELO (đã filter ở preprocessing nhưng kiểm tra lại)
null_counts = df_sample.null_count()
print(null_counts)

# ECO code bất thường
eco_invalid = df_sample.filter(~pl.col("ECO").str.contains(r"^[A-E]\d{2}$"))
print(f"ECO không hợp lệ: {len(eco_invalid)} rows")
```

## Integration Points
**Kết nối giữa các thành phần**

- `src/config.py`: Tất cả paths (DATA_PROCESSED, EDA_OUTPUTS) lấy từ config. Không hardcode.
- Parquet files: Output từ pipeline preprocessing (Giai đoạn 1) → input cho EDA
- Kết quả EDA (insights, feature ranking) → input cho Feature Engineering (Giai đoạn 3)

## Security Notes
**Bảo mật**

> N/A — EDA notebook chạy local, không expose data hay credentials. Dữ liệu Lichess là public domain.

## Performance Considerations
**Cách giữ notebook nhanh**

### CPU (Polars 20 threads)
- **Sampling**: Luôn sample trước khi plot (3M rows max cho visualization)
- **Lazy eval**: Dùng `scan_parquet()` thay `read_parquet()` cho aggregation
- **Column selection**: Chỉ select cột cần thiết (`pl.exclude("Moves")` khi không cần parse moves)
- **String operations**: Dùng `str.split()` thay vì `python-chess` cho move parsing khi có thể
- **Plot size**: Limit data points trong scatter plots (sample hoặc hexbin)
- **Memory cleanup**: `del df_temp` và `gc.collect()` sau các operations lớn

### GPU (RTX 3060 12GB)
- **XGBoost GPU**: Dùng `device='cuda'` cho feature importance trên sample 1-3M rows (~30s thay vì 5+ phút CPU)
- **VRAM budget**: 12 GB VRAM — sample 3M rows với ~10 features sử dụng ~2-4 GB VRAM, dư thoải mái
- **Fallback**: Nếu GPU fail (driver issue, OOM), tự động fallback về CPU với try/except
- **Không dùng GPU cho**: Data wrangling (Polars CPU đã đủ nhanh), visualization (CPU-bound)
