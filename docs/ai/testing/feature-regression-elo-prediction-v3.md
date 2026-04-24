---
phase: testing
title: Testing V3 — Chuyển từ Classification sang Regression dự đoán ELO liên tục
description: Chiến lược kiểm thử cho eval_xgboost_v3.py — đảm bảo tính đúng đắn và chất lượng model
---

# Testing V3 — Regression ELO Prediction

## Test Coverage Goals

- Unit test: 100% các function trong `eval_xgboost_v3.py`
- Integration test: End-to-end pipeline (load → train → predict → plot → save)
- Validation: Metrics đạt acceptance criteria (MAE ≤ 220, R² ≥ 0.55)
- Cross-check: So sánh ngược với V2 Classification (47.59%)

## Unit Tests

### Module: Data Loading (`load_data`)
- [ ] **Test 1.1**: Load features V2 thành công — verify shape (30000, ~117)
- [ ] **Test 1.2**: Load raw data thành công — verify `EloAvg` column tồn tại
- [ ] **Test 1.3**: Join X, y khớp shape — `len(X) == len(y) == 30000`
- [ ] **Test 1.4**: Target y không có NaN — `np.isnan(y).sum() == 0`
- [ ] **Test 1.5**: Target y range hợp lý — `y.min() >= 200`, `y.max() <= 4000`
- [ ] **Test 1.6**: Target y stats — `mean ≈ 1650 ± 200`, `std ≈ 400 ± 100`

### Module: Model Training (`run_regression_cv`)
- [ ] **Test 2.1**: Trả về đúng 5 fold results (list 5 dicts)
- [ ] **Test 2.2**: Mỗi fold result có keys: `fold`, `mae`, `rmse`, `r2`
- [ ] **Test 2.3**: MAE > 0 cho mỗi fold (sanity check)
- [ ] **Test 2.4**: R² > 0 cho mỗi fold (model tốt hơn mean predictor)
- [ ] **Test 2.5**: `all_y_true` + `all_y_pred` có length = 30000

### Module: Ablation Study (`run_ablation_study`)
- [ ] **Test 3.1**: Trả về đúng 4 configs (A/B/C/D)
- [ ] **Test 3.2**: MAE giảm dần từ Config A → Config D (monotonically)
- [ ] **Test 3.3**: Config D (Full) cho MAE thấp nhất

### Module: Plotting
- [ ] **Test 4.1**: `scatter_actual_vs_predicted.png` được tạo ra, filesize > 0
- [ ] **Test 4.2**: `residual_distribution.png` được tạo ra, filesize > 0
- [ ] **Test 4.3**: `mae_by_elo_band.png` được tạo ra, filesize > 0
- [ ] **Test 4.4**: `feature_importance_v3.png` được tạo ra, filesize > 0

### Module: Classification Comparison
- [ ] **Test 5.1**: `compare_with_classification` trả về accuracy (float 0-1)
- [ ] **Test 5.2**: Accuracy > 0.40 (ít nhất tốt hơn random)
- [ ] **Test 5.3**: Classification report có đủ 5 bands

### Module: Results Saving
- [ ] **Test 6.1**: `eval_results_v3.json` được tạo ra
- [ ] **Test 6.2**: JSON parseable — `json.load()` không lỗi
- [ ] **Test 6.3**: JSON chứa keys: `regression_metrics`, `ablation`, `classification_comparison`

## Integration Tests

### End-to-End Pipeline
- [ ] **E2E-1**: Chạy `python src/eval_xgboost_v3.py` hoàn tất không lỗi
- [ ] **E2E-2**: Tất cả 5 output files tồn tại trong `data/results/v3/`
- [ ] **E2E-3**: Runtime < 15 phút

### Cross-Validation Consistency
- [ ] **CV-1**: Chạy lại lần 2 → cùng kết quả (reproducibility qua random_state=42)
- [ ] **CV-2**: MAE std across folds < 20 ELO (model ổn định)

## Acceptance Criteria Validation

| Metric | Mục tiêu | Test |
|--------|----------|------|
| MAE | ≤ 220 ELO | `assert mean_mae <= 220` |
| RMSE | ≤ 280 ELO | `assert mean_rmse <= 280` |
| R² | ≥ 0.55 | `assert mean_r2 >= 0.55` |
| Regression→Class Accuracy | ≥ 50% | `assert class_accuracy >= 0.50` |
| Ablation monotonicity | A > B > C > D (MAE) | `assert mae_A > mae_B > mae_C > mae_D` |

## Test Data

- **Primary**: `data/features/sample_30k_features_v2.parquet` + `data/processed/sample_30k.parquet`
- **Smoke test**: Có thể tạo subset 1000 ván cho test nhanh
- **Không cần mock**: Dùng real data vì file nhỏ (< 2MB) và load nhanh

## Manual Testing

### Checklist trực quan
- [ ] Scatter plot: Đám mây điểm bám sát đường y=x?
- [ ] Residual: Phân phối chuẩn, tập trung quanh 0?
- [ ] MAE by Band: Beginner/Master thấp, Advanced cao? (pattern kỳ vọng)
- [ ] Feature Importance: `avg_wdl_loss`, `best_move_match_rate` nằm top?

### So sánh V2 vs V3
- [ ] V3 Regression→Class ≥ V2 Classification (47.59%)?
- [ ] Bảng so sánh in ra rõ ràng, dễ đọc?

## Performance Testing

| Test | Target | Ghi chú |
|------|--------|---------|
| Single fold training time | < 30s | 300 trees trên 24k samples |
| Full 5-fold CV | < 3 phút | 5 × 30s |
| Full ablation (4×5 folds) | < 12 phút | Pipeline hoàn chỉnh |
| Plot generation | < 1 phút | 4 biểu đồ matplotlib |

## Bug Tracking

### Các lỗi tiềm ẩn cần chú ý
1. **Index mismatch**: Nếu features V2 và raw parquet có thứ tự index khác nhau → EloAvg gán sai
2. **NaN propagation**: NaN trong features (midgame_cpl, endgame_cpl) → XGBoost xử lý ok nhưng cần verify
3. **ELO bins edge cases**: ELO = 1000, 1400, 1800, 2200 → rơi vào bin nào? (check `pd.cut` behavior: right=True by default)
4. **Prediction outliers**: Model có thể predict ELO < 0 hoặc > 4000 → cần clip hoặc ghi nhận
