---
phase: implementation
title: Implementation V3 — Chuyển từ Classification sang Regression dự đoán ELO liên tục
description: Hướng dẫn kỹ thuật chi tiết cho việc triển khai eval_xgboost_v3.py
---

# Implementation V3 — Regression ELO Prediction

## Development Setup

### Prerequisites
- Python 3.10+ (conda environment đã cấu hình)
- XGBoost >= 1.7 (đã cài trong environment)
- scikit-learn, pandas, numpy, matplotlib (đã cài)

### Cấu hình
```bash
# Kích hoạt environment
conda activate

# Verify dependencies
python -c "import xgboost; print(xgboost.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

### Dữ liệu cần thiết
```
data/features/sample_30k_features_v2.parquet   # ~1.1MB, 30k rows × ~117 cols
data/processed/sample_30k.parquet               # Chứa EloAvg target column
```

## Code Structure

### Cấu trúc file `src/eval_xgboost_v3.py`
```
eval_xgboost_v3.py
├── Imports & Constants
│   ├── XGB_REG_PARAMS            # Hyperparameters XGBRegressor
│   ├── FEATURE_GROUPS            # 4 nhóm features cho Ablation
│   └── ELO_BINS / BAND_LABELS   # Bins chuyển ELO → Class
│
├── load_data()
│   ├── Load features V2 parquet
│   ├── Load raw parquet → lấy EloAvg
│   └── Return X, y
│
├── run_regression_cv(X, y, params, n_splits=5)
│   ├── KFold split
│   ├── Train/predict mỗi fold
│   ├── Tính MAE, RMSE, R² mỗi fold
│   └── Return results dict + all predictions
│
├── run_ablation_study(X, y, feature_groups)
│   ├── Chạy CV cho mỗi config (A/B/C/D)
│   └── Return ablation_results dict
│
├── plot_scatter(y_true, y_pred, output_path)
├── plot_residuals(y_true, y_pred, output_path)
├── plot_mae_by_band(y_true, y_pred, output_path)
├── plot_feature_importance(model, feature_names, output_path)
│
├── compare_with_classification(y_true, y_pred)
│   ├── Chuyển ELO → Band
│   └── Tính accuracy + classification report
│
├── save_results(results, output_path)
│
└── main()
    ├── load_data()
    ├── run_regression_cv()
    ├── run_ablation_study()
    ├── plot_*() × 4
    ├── compare_with_classification()
    └── save_results()
```

## Implementation Notes

### Core: Load Data & Join Target
```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features" / "sample_30k_features_v2.parquet"
RAW_PATH = DATA_DIR / "processed" / "sample_30k.parquet"
OUTPUT_DIR = DATA_DIR / "results" / "v3"

def load_data():
    """Load features V2 và join EloAvg target từ raw data."""
    features_df = pd.read_parquet(FEATURES_PATH)
    raw_df = pd.read_parquet(RAW_PATH)
    
    # Lấy target
    y = raw_df["EloAvg"].values.astype(np.float32)
    
    # Features: bỏ ModelBand (classification target)
    X = features_df.drop(columns=["ModelBand"], errors="ignore")
    
    assert len(X) == len(y), f"Shape mismatch: X={len(X)}, y={len(y)}"
    print(f"Loaded: X={X.shape}, y range=[{y.min():.0f}, {y.max():.0f}], mean={y.mean():.0f}")
    
    return X, y
```

### Core: XGBRegressor + Cross-Validation
```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

XGB_REG_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}

def run_regression_cv(X, y, params=XGB_REG_PARAMS, n_splits=5):
    """Chạy KFold CV cho XGBRegressor, trả về metrics + predictions."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    all_y_true, all_y_pred = [], []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        fold_results.append({"fold": fold, "mae": mae, "rmse": rmse, "r2": r2})
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        
        print(f"  Fold {fold}: MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.4f}")
    
    return fold_results, np.array(all_y_true), np.array(all_y_pred), model
```

### Core: Ablation Study Feature Groups
```python
# Nhóm features cho Ablation (giống cấu trúc V2)
ENGINE_GROUP_A = ["avg_cpl", "cpl_std", "blunder_rate", "mistake_rate", "inaccuracy_rate"]
ENGINE_GROUP_B = ["opening_cpl", "midgame_cpl", "endgame_cpl"]
ENGINE_GROUP_C = ["avg_wdl_loss", "max_wdl_loss", "best_move_match_rate"]

# Tabular features = tất cả cột không thuộc engine groups
TABULAR_COLS = [c for c in X.columns if c not in ENGINE_GROUP_A + ENGINE_GROUP_B + ENGINE_GROUP_C]

ABLATION_CONFIGS = {
    "A: Tabular Only": TABULAR_COLS,
    "B: Tabular + Group A": TABULAR_COLS + ENGINE_GROUP_A,
    "C: Tabular + A + B": TABULAR_COLS + ENGINE_GROUP_A + ENGINE_GROUP_B,
    "D: Full V2": TABULAR_COLS + ENGINE_GROUP_A + ENGINE_GROUP_B + ENGINE_GROUP_C,
}
```

### Core: So sánh ngược Classification
```python
ELO_BINS = [0, 1000, 1400, 1800, 2200, 9999]
BAND_NAMES = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]

def compare_with_classification(y_true, y_pred):
    """Chuyển ELO → Band, tính accuracy/f1 để đối chiếu V2."""
    true_bands = pd.cut(y_true, bins=ELO_BINS, labels=BAND_NAMES)
    pred_bands = pd.cut(y_pred, bins=ELO_BINS, labels=BAND_NAMES)
    
    accuracy = (true_bands == pred_bands).mean()
    
    from sklearn.metrics import classification_report
    report = classification_report(true_bands, pred_bands, output_dict=True)
    
    print(f"\n=== Regression → Classification (đối chiếu V2: 47.59%) ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(classification_report(true_bands, pred_bands))
    
    return accuracy, report
```

## Patterns & Best Practices

1. **File standalone**: `eval_xgboost_v3.py` không import từ `eval_xgboost_v2.py` — tránh coupling
2. **Reproducibility**: Mọi random seed = 42
3. **Output directory**: Tự tạo `data/results/v3/` nếu chưa có (`os.makedirs(exist_ok=True)`)
4. **Logging**: Print progress cho mỗi fold + mỗi ablation config
5. **Type hints**: Sử dụng type hints cho các function signatures

## Error Handling

- **File not found**: Check paths trước khi load, print message rõ ràng
- **Shape mismatch**: Assert `len(X) == len(y)` sau khi join
- **NaN in target**: Verify `y` không có NaN (model không train được với NaN target)
- **NaN in features**: XGBoost xử lý NaN natively → không cần xử lý
- **Prediction outliers**: Clip predictions vào range [400, 3700] nếu cần (optional)

## Performance Considerations

- Training 300 trees × 5 folds × 30k samples: **< 3 phút** trên i5-14600KF
- Ablation 4 configs × 5 folds: **< 10 phút** tổng
- Plotting 4 biểu đồ: **< 1 phút**
- **Tổng runtime**: ~15 phút (bao gồm hết)
