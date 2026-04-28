"""Đánh giá XGBoost Regression V3 — Dự đoán ELO liên tục.

Theo Planning V3:
- Chuyển từ Classification (5 lớp) sang Regression (ELO liên tục)
- Tái sử dụng 11 features Stockfish V2
- Target: EloAvg (Float) thay vì ModelBand (Int 0-4)
- Model: XGBRegressor thay vì XGBClassifier
- Metrics: MAE / RMSE / R² thay vì Accuracy / F1

Triết lý: Giữ nguyên Lò Bát Quái (Features V2), chỉ thay đổi Phép Đo.
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend cho SSH
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Cấu hình ───────────────────────────────────────────────
V2_FEATURES_PATH = "data/features/sample_30k_features_v2.parquet"
RAW_DATA_PATH = "data/processed/sample_30k.parquet"
OUTPUT_DIR = Path("data/results/v3")

# V2 Classification baseline để so sánh
V2_ACCURACY = 47.59

# Nhóm features engine V2 (giữ nguyên từ V2)
GROUP_A = ["avg_cpl", "cpl_std", "blunder_rate", "mistake_rate", "inaccuracy_rate"]
GROUP_B = ["opening_cpl", "midgame_cpl", "endgame_cpl"]
GROUP_C = ["avg_wdl_loss", "max_wdl_loss", "best_move_match_rate"]
ALL_ENGINE = GROUP_A + GROUP_B + GROUP_C

# Bins chuyển ELO → Band (giống V2 Classification)
ELO_BINS = [0, 1000, 1400, 1800, 2200, 9999]
BAND_NAMES = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]

# Tham số XGBRegressor
XGB_REG_PARAMS = {
    "objective": "reg:squarederror",   # Loss function: MSE
    "eval_metric": "mae",              # Đánh giá bằng MAE
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,               # Tăng từ 200 (V2) vì regression cần nhiều hơn
    "random_state": 42,
    "n_jobs": -1,
}


# ── Load Data ──────────────────────────────────────────────

def load_data() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Load features V2 và join EloAvg target từ raw data.

    Returns:
        features_df: DataFrame chứa features (bỏ ModelBand)
        X: numpy array features
        y: numpy array EloAvg target
    """
    # Load features V2
    features_df = pl.read_parquet(V2_FEATURES_PATH)
    print(f"  Features V2: {features_df.shape[0]} ván × {features_df.shape[1]} cột")

    # Load raw data để lấy EloAvg
    raw_df = pl.read_parquet(RAW_DATA_PATH)
    y = raw_df["EloAvg"].to_numpy().astype(np.float32)

    # Tách features (bỏ ModelBand — classification target)
    feature_cols = [c for c in features_df.columns if c != "ModelBand"]
    X = features_df.select(feature_cols).to_numpy()

    # Verify
    assert len(X) == len(y), f"Shape mismatch: X={len(X)}, y={len(y)}"
    assert not np.isnan(y).any(), "Target EloAvg chứa NaN!"

    print(f"  Target EloAvg: range=[{y.min():.0f}, {y.max():.0f}], "
          f"mean={y.mean():.0f}, std={y.std():.0f}")
    print(f"  Features: {X.shape[1]} cột (bỏ ModelBand)")

    return features_df, feature_cols, X, y


# ── Cross-Validation Regression ────────────────────────────

def run_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
    name: str,
    params: dict = None,
    n_splits: int = 5,
) -> dict:
    """Chạy KFold CV cho XGBRegressor, trả về metrics + predictions.

    Args:
        X: Feature matrix
        y: Target EloAvg
        feat_names: Tên các features
        name: Tên config (cho logging)
        params: Hyperparameters cho XGBRegressor
        n_splits: Số folds

    Returns:
        Dict chứa metrics, predictions, importances
    """
    if params is None:
        params = XGB_REG_PARAMS

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_maes, fold_rmses, fold_r2s = [], [], []
    oof_preds = np.zeros(len(y), dtype=np.float32)
    importances = np.zeros(X.shape[1])
    last_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds = model.predict(X_va)
        oof_preds[val_idx] = preds

        mae = mean_absolute_error(y_va, preds)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        r2 = r2_score(y_va, preds)

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        importances += model.feature_importances_ / n_splits
        last_model = model

        print(f"    Fold {fold}: MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.4f}")

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    mean_r2 = np.mean(fold_r2s)
    std_r2 = np.std(fold_r2s)

    print(f"\n  [{name}]")
    print(f"  * MAE  : {mean_mae:.1f} ELO (±{std_mae:.1f})")
    print(f"  * RMSE : {mean_rmse:.1f} ELO (±{std_rmse:.1f})")
    print(f"  * R²   : {mean_r2:.4f} (±{std_r2:.4f})")

    return {
        "name": name,
        "mae_mean": round(float(mean_mae), 2),
        "mae_std": round(float(std_mae), 2),
        "rmse_mean": round(float(mean_rmse), 2),
        "rmse_std": round(float(std_rmse), 2),
        "r2_mean": round(float(mean_r2), 4),
        "r2_std": round(float(std_r2), 4),
        "oof_preds": oof_preds,
        "importances": importances,
        "feat_names": feat_names,
        "model": last_model,
    }


# ── Trực quan hóa ─────────────────────────────────────────

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, r2: float, output_path: Path):
    """Scatter Plot: Actual vs Predicted ELO."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot với alpha thấp vì 30k điểm
    ax.scatter(y_true, y_pred, alpha=0.1, s=5, c="steelblue", edgecolors="none")

    # Đường y=x (perfect prediction)
    elo_min, elo_max = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([elo_min, elo_max], [elo_min, elo_max], 'r--', linewidth=2, label="Perfect (y=x)")

    # Annotations
    mae = mean_absolute_error(y_true, y_pred)
    ax.set_title(f"Actual vs Predicted ELO — XGBRegressor V3\n"
                 f"R²={r2:.4f}, MAE={mae:.1f} ELO", fontsize=14, fontweight="bold")
    ax.set_xlabel("ELO thực (EloAvg)", fontsize=12)
    ax.set_ylabel("ELO dự đoán", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✅ Scatter plot saved: {output_path}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Residual Distribution: Histogram (Predicted - Actual)."""
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=80, color="steelblue", alpha=0.7, edgecolor="white")

    # Đường mean
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax.axvline(mean_res, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {mean_res:.1f}")
    ax.axvline(mean_res + std_res, color="orange", linestyle=":", linewidth=1.5,
               label=f"±1σ = {std_res:.1f}")
    ax.axvline(mean_res - std_res, color="orange", linestyle=":", linewidth=1.5)

    ax.set_title("Phân phối Sai số (Residuals) — V3 Regression\n"
                 f"Mean={mean_res:.1f}, Std={std_res:.1f} ELO", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sai số (Predicted - Actual) ELO", fontsize=12)
    ax.set_ylabel("Số lượng ván", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✅ Residual distribution saved: {output_path}")


def plot_mae_by_band(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """MAE phân theo ELO Band: Bar chart 5 dải."""
    import pandas as pd

    # Gán band cho y_true
    bands = pd.cut(y_true, bins=ELO_BINS, labels=BAND_NAMES)
    errors = np.abs(y_pred - y_true)

    # Tính MAE trung bình mỗi band
    band_maes = {}
    band_counts = {}
    for band_name in BAND_NAMES:
        mask = np.asarray(bands == band_name)
        if mask.sum() > 0:
            band_maes[band_name] = float(errors[mask].mean())
            band_counts[band_name] = int(mask.sum())
        else:
            band_maes[band_name] = 0
            band_counts[band_name] = 0

    # Vẽ bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#f39c12", "#2ecc71"]  # Xanh-Cam-Đỏ-Cam-Xanh
    bars = ax.bar(BAND_NAMES, [band_maes[b] for b in BAND_NAMES], color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5)

    # Annotate giá trị MAE lên mỗi bar
    for bar, band in zip(bars, BAND_NAMES):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 3,
                f'{height:.0f}\n(n={band_counts[band]})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title("MAE phân theo ELO Band — V3 Regression", fontsize=14, fontweight="bold")
    ax.set_xlabel("ELO Band", fontsize=12)
    ax.set_ylabel("MAE (ELO)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✅ MAE by band saved: {output_path}")

    return band_maes


def plot_feature_importance(importances: np.ndarray, feat_names: list[str], output_path: Path):
    """Top 20 Feature Importance plot."""
    top_idx = np.argsort(importances)[-20:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        np.array(feat_names)[top_idx],
        importances[top_idx],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_title("Top 20 Feature Importances — V3 Regression (XGBRegressor)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✅ Feature importance saved: {output_path}")


# ── So sánh ngược V2 Classification ────────────────────────

def compare_with_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Chuyển ELO dự đoán → Band, tính accuracy/f1 để đối chiếu V2.

    Returns:
        Dict chứa accuracy, macro_f1, classification report
    """
    import pandas as pd

    # Chuyển ELO → Band
    true_bands = pd.cut(y_true, bins=ELO_BINS, labels=BAND_NAMES).astype(str)
    pred_bands = pd.cut(
        np.clip(y_pred, ELO_BINS[0] + 1, ELO_BINS[-1] - 1),  # Clip outliers
        bins=ELO_BINS,
        labels=BAND_NAMES,
    ).astype(str)

    accuracy = accuracy_score(true_bands, pred_bands)
    report_dict = classification_report(true_bands, pred_bands, output_dict=True)
    report_str = classification_report(true_bands, pred_bands)
    macro_f1 = report_dict["macro avg"]["f1-score"]

    print(f"\n{'═' * 60}")
    print(f"  SO SÁNH NGƯỢC: REGRESSION → CLASSIFICATION")
    print(f"{'─' * 60}")
    print(f"  V2 Classification trực tiếp : {V2_ACCURACY:.2f}%")
    print(f"  V3 Regression → Class       : {accuracy * 100:.2f}%")
    delta = accuracy * 100 - V2_ACCURACY
    print(f"  Delta                       : {'+' if delta > 0 else ''}{delta:.2f}%")
    print(f"  Macro F1                    : {macro_f1 * 100:.2f}%")
    print(f"\n{report_str}")

    # Confusion Matrix
    cm = confusion_matrix(true_bands, pred_bands, labels=BAND_NAMES)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=BAND_NAMES, yticklabels=BAND_NAMES, ax=ax)
    ax.set_title("Confusion Matrix — V3 Regression → Classification", fontsize=14)
    ax.set_xlabel("Predicted Band")
    ax.set_ylabel("Actual Band")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix_v3.png"
    fig.savefig(str(cm_path), dpi=150)
    plt.close(fig)
    print(f"  ✅ Confusion matrix saved: {cm_path}")

    return {
        "accuracy": round(float(accuracy * 100), 2),
        "macro_f1": round(float(macro_f1 * 100), 2),
        "v2_accuracy": V2_ACCURACY,
        "delta": round(float(delta), 2),
    }


# ── Main ───────────────────────────────────────────────────

def main():
    print(f"{'═' * 60}")
    print("  ĐÁNH GIÁ XGBOOST REGRESSION V3 — DỰ ĐOÁN ELO LIÊN TỤC")
    print(f"  Triết lý: Giữ nguyên Features V2, chỉ thay Phép Đo")
    print(f"{'═' * 60}")

    # ── 1. Load data ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 1: LOAD DATA")
    print(f"{'─' * 60}")
    features_df, feature_cols, X, y = load_data()

    # Phân chia features cho ablation study
    tabular_features = [c for c in feature_cols if c not in ALL_ENGINE]
    tab_a = tabular_features + GROUP_A
    tab_ab = tabular_features + GROUP_A + GROUP_B
    tab_abc = tabular_features + ALL_ENGINE  # = tất cả features

    # ── 2. Full Model Regression ─────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 2: FULL MODEL — XGBRegressor 5-Fold CV")
    print(f"{'─' * 60}")

    full_result = run_regression_cv(X, y, feature_cols, "D: Full V2 (A+B+C)")

    # ── 3. Ablation Study ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 3: ABLATION STUDY")
    print(f"{'─' * 60}")

    ablation_results = []

    # Config A: Tabular Only
    print("\n  [A] Tabular Only...")
    X_tab = features_df.select(tabular_features).to_numpy()
    res_a = run_regression_cv(X_tab, y, tabular_features, "A: Tabular Only")
    ablation_results.append(res_a)

    # Config B: Tabular + Nhóm A
    print("\n  [B] Tabular + Nhóm A (CPL tỷ lệ hóa)...")
    X_tab_a = features_df.select(tab_a).to_numpy()
    res_b = run_regression_cv(X_tab_a, y, tab_a, "B: Tabular + Nhóm A")
    ablation_results.append(res_b)

    # Config C: Tabular + Nhóm A + B
    print("\n  [C] Tabular + Nhóm A + Nhóm B (Phase CPL)...")
    X_tab_ab = features_df.select(tab_ab).to_numpy()
    res_c = run_regression_cv(X_tab_ab, y, tab_ab, "C: Tabular + A + B")
    ablation_results.append(res_c)

    # Config D đã chạy ở Phase 2
    ablation_results.append(full_result)

    # Bảng tổng hợp Ablation
    print(f"\n{'═' * 60}")
    print("  ABLATION SUMMARY")
    print(f"{'─' * 60}")
    print(f"  {'Config':<30s} {'MAE':>10s} {'RMSE':>10s} {'R²':>10s}")
    print(f"  {'─' * 60}")
    for r in ablation_results:
        print(
            f"  {r['name']:<30s} "
            f"{r['mae_mean']:>6.1f}±{r['mae_std']:<4.1f} "
            f"{r['rmse_mean']:>6.1f}±{r['rmse_std']:<4.1f} "
            f"{r['r2_mean']:>6.4f}±{r['r2_std']:<6.4f}"
        )

    # ── 4. Trực quan hóa ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 4: TRỰC QUAN HÓA")
    print(f"{'─' * 60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_pred_oof = full_result["oof_preds"]
    r2_val = full_result["r2_mean"]

    plot_scatter(y, y_pred_oof, r2_val, OUTPUT_DIR / "scatter_actual_vs_predicted.png")
    plot_residuals(y, y_pred_oof, OUTPUT_DIR / "residual_distribution.png")
    band_maes = plot_mae_by_band(y, y_pred_oof, OUTPUT_DIR / "mae_by_elo_band.png")
    plot_feature_importance(
        full_result["importances"],
        full_result["feat_names"],
        OUTPUT_DIR / "feature_importance_v3.png",
    )

    # ── 5. So sánh ngược V2 ─────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 5: SO SÁNH NGƯỢC V2 CLASSIFICATION")
    print(f"{'─' * 60}")

    class_comparison = compare_with_classification(y, y_pred_oof)

    # ── 6. Lưu kết quả ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 6: LƯU KẾT QUẢ")
    print(f"{'─' * 60}")

    summary = {
        "model": "XGBRegressor",
        "version": "V3",
        "target": "EloAvg (regression liên tục)",
        "data": {
            "features_file": V2_FEATURES_PATH,
            "raw_data_file": RAW_DATA_PATH,
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "target_range": [float(y.min()), float(y.max())],
            "target_mean": round(float(y.mean()), 1),
            "target_std": round(float(y.std()), 1),
        },
        "params": XGB_REG_PARAMS,
        "regression_metrics": {
            "mae_mean": full_result["mae_mean"],
            "mae_std": full_result["mae_std"],
            "rmse_mean": full_result["rmse_mean"],
            "rmse_std": full_result["rmse_std"],
            "r2_mean": full_result["r2_mean"],
            "r2_std": full_result["r2_std"],
        },
        "ablation": [
            {
                "config": r["name"],
                "mae_mean": r["mae_mean"],
                "mae_std": r["mae_std"],
                "rmse_mean": r["rmse_mean"],
                "rmse_std": r["rmse_std"],
                "r2_mean": r["r2_mean"],
                "r2_std": r["r2_std"],
            }
            for r in ablation_results
        ],
        "mae_by_band": {k: round(v, 2) for k, v in band_maes.items()},
        "classification_comparison": class_comparison,
    }

    json_path = OUTPUT_DIR / "eval_results_v3.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Results JSON saved: {json_path}")

    # ── 7. Báo cáo tổng kết ─────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  📊 BÁO CÁO TỔNG KẾT V3 REGRESSION")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<30s} {'V3 Regression':>15s}")
    print(f"  {'─' * 45}")
    print(f"  {'MAE (ELO)':<30s} {full_result['mae_mean']:>10.1f} ±{full_result['mae_std']:.1f}")
    print(f"  {'RMSE (ELO)':<30s} {full_result['rmse_mean']:>10.1f} ±{full_result['rmse_std']:.1f}")
    print(f"  {'R² Score':<30s} {full_result['r2_mean']:>10.4f} ±{full_result['r2_std']:.4f}")
    print(f"  {'─' * 45}")
    print(f"  {'V2 Classification Acc':<30s} {V2_ACCURACY:>10.2f}%")
    print(f"  {'V3 Regression→Class Acc':<30s} {class_comparison['accuracy']:>10.2f}%")
    print(f"  {'Delta':<30s} {'+' if class_comparison['delta'] > 0 else ''}{class_comparison['delta']:>10.2f}%")

    # Đánh giá kết quả
    mae_val = full_result["mae_mean"]
    if mae_val <= 180:
        verdict = "🟢 ĐẠI THẮNG! MAE ≤ 180 — Vượt kỳ vọng"
    elif mae_val <= 220:
        verdict = "🟢 THẮNG! MAE ≤ 220 — Đạt mục tiêu"
    elif mae_val <= 250:
        verdict = "🟡 Chấp nhận được. MAE ≤ 250 — Gần mục tiêu"
    else:
        verdict = "🔴 Cần cải thiện. MAE > 250 — Cân nhắc V4"

    print(f"\n  {'─' * 45}")
    print(f"  ĐÁNH GIÁ: {verdict}")

    print(f"\n{'═' * 60}")
    print("  ✅ HOÀN THÀNH ĐÁNH GIÁ V3 REGRESSION!")
    print(f"  📁 Output: {OUTPUT_DIR}/")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
