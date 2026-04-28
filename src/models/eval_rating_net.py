"""Đánh giá Rating Net (MLP) V4 — Dự đoán ELO liên tục bằng PyTorch.

Theo yêu cầu nhóm trưởng:
- Bỏ LSTM/clock, chạy thử rating net với CPL + blunder
- Dùng 11 aggregate features V2 có sẵn (không extract sequence mới)
- MLP feedforward đơn giản: 256 → 128 → 64 → 1
- So sánh với XGBoost V3 baseline (MAE 247.81)

Kiến trúc:
- Input: 117 features (hoặc subset cho ablation)
- Hidden: Linear(256) → ReLU → Dropout(0.3) → Linear(128) → ReLU → Dropout(0.2) → Linear(64) → ReLU
- Output: 1 node (EloAvg regression)
- Loss: MSE, Optimizer: Adam, Scheduler: ReduceLROnPlateau
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Cấu hình ───────────────────────────────────────────────
V2_FEATURES_PATH = "data/features/sample_30k_features_v2.parquet"
RAW_DATA_PATH = "data/processed/sample_30k.parquet"
OUTPUT_DIR = Path("data/results/v4")

# Nhóm features engine V2 (giữ nguyên từ V2/V3)
GROUP_A = ["avg_cpl", "cpl_std", "blunder_rate", "mistake_rate", "inaccuracy_rate"]
GROUP_B = ["opening_cpl", "midgame_cpl", "endgame_cpl"]
GROUP_C = ["avg_wdl_loss", "max_wdl_loss", "best_move_match_rate"]
ALL_ENGINE = GROUP_A + GROUP_B + GROUP_C

# Tham số training
BATCH_SIZE = 512
EPOCHS = 200
LR = 1e-3
PATIENCE = 15  # Early stopping patience
N_SPLITS = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ──────────────────────────────────────────────────

class RatingNet(nn.Module):
    """MLP feedforward cho ELO rating prediction."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Load Data ──────────────────────────────────────────────

def load_data() -> tuple[pl.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Load features V2 và join EloAvg target từ raw data."""
    features_df = pl.read_parquet(V2_FEATURES_PATH)
    print(f"  Features V2: {features_df.shape[0]} ván × {features_df.shape[1]} cột")

    raw_df = pl.read_parquet(RAW_DATA_PATH)
    y = raw_df["EloAvg"].to_numpy().astype(np.float32)

    feature_cols = [c for c in features_df.columns if c != "ModelBand"]
    X = features_df.select(feature_cols).to_numpy().astype(np.float32)

    # Fill NaN (midgame_cpl/endgame_cpl cho ván ngắn) bằng 0
    X = np.nan_to_num(X, nan=0.0)

    assert len(X) == len(y), f"Shape mismatch: X={len(X)}, y={len(y)}"
    assert not np.isnan(y).any(), "Target EloAvg chứa NaN!"

    print(f"  Target EloAvg: range=[{y.min():.0f}, {y.max():.0f}], "
          f"mean={y.mean():.0f}, std={y.std():.0f}")
    print(f"  Features: {X.shape[1]} cột (bỏ ModelBand)")
    print(f"  NaN đã fill 0: {np.isnan(X).sum()} remaining")
    print(f"  Device: {DEVICE}")

    return features_df, feature_cols, X, y


# ── Training Loop ──────────────────────────────────────────

def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    fold: int,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Train 1 fold, trả về predictions + loss curves."""
    # Normalize features (fit trên train, transform cả train+val)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_va_scaled = scaler.transform(X_val).astype(np.float32)

    # Tạo DataLoader
    train_ds = TensorDataset(
        torch.from_numpy(X_tr_scaled),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_va_scaled),
        torch.from_numpy(y_val),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = RatingNet(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(X_batch)
        epoch_train_loss /= len(train_ds)

        # ── Val ──
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                epoch_val_loss += loss.item() * len(X_batch)
        epoch_val_loss /= len(val_ds)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Fold {fold}: Early stop tại epoch {epoch + 1}, "
                      f"best val_loss={best_val_loss:.2f}")
                break

    # Load best model, predict val
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())

    val_preds = np.concatenate(all_preds)
    return val_preds, train_losses, val_losses


# ── Cross-Validation ───────────────────────────────────────

def run_mlp_cv(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
    name: str,
) -> dict:
    """Chạy KFold CV cho RatingNet MLP, trả về metrics."""
    print(f"\n  [{name}] — {len(feat_names)} features")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_maes, fold_rmses, fold_r2s = [], [], []
    oof_preds = np.zeros(len(y), dtype=np.float32)
    all_train_losses = []
    all_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        val_preds, train_losses, val_losses = train_fold(
            X_tr, y_tr, X_va, y_va, X.shape[1], fold,
        )
        oof_preds[val_idx] = val_preds

        mae = mean_absolute_error(y_va, val_preds)
        rmse = np.sqrt(mean_squared_error(y_va, val_preds))
        r2 = r2_score(y_va, val_preds)

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        print(f"    Fold {fold}: MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.4f} "
              f"(epochs={len(train_losses)})")

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    mean_r2 = np.mean(fold_r2s)
    std_r2 = np.std(fold_r2s)

    print(f"\n  [{name}] Kết quả:")
    print(f"  * MAE  : {mean_mae:.1f} ELO (±{std_mae:.1f})")
    print(f"  * RMSE : {mean_rmse:.1f} ELO (±{std_rmse:.1f})")
    print(f"  * R²   : {mean_r2:.4f} (±{std_r2:.4f})")

    return {
        "name": name,
        "n_features": len(feat_names),
        "mae_mean": round(float(mean_mae), 2),
        "mae_std": round(float(std_mae), 2),
        "rmse_mean": round(float(mean_rmse), 2),
        "rmse_std": round(float(std_rmse), 2),
        "r2_mean": round(float(mean_r2), 4),
        "r2_std": round(float(std_r2), 4),
        "oof_preds": oof_preds,
        "train_losses": all_train_losses,
        "val_losses": all_val_losses,
    }


# ── Trực quan hóa ─────────────────────────────────────────

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, output_path: Path):
    """Scatter Plot: Actual vs Predicted ELO."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, y_pred, alpha=0.1, s=5, c="steelblue", edgecolors="none")

    elo_min = min(y_true.min(), y_pred.min())
    elo_max = max(y_true.max(), y_pred.max())
    ax.plot([elo_min, elo_max], [elo_min, elo_max], 'r--', linewidth=2, label="Perfect (y=x)")

    ax.set_title(f"Actual vs Predicted ELO — RatingNet MLP V4\n"
                 f"R²={metrics['r2_mean']:.4f}, MAE={metrics['mae_mean']:.1f} ELO",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("ELO thực (EloAvg)", fontsize=12)
    ax.set_ylabel("ELO dự đoán", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✅ Scatter plot saved: {output_path}")


def plot_loss_curves(train_losses: list, val_losses: list, output_path: Path):
    """Training/Validation loss curves trung bình qua các folds."""
    # Pad tất cả folds về cùng độ dài (lấy max epochs)
    max_epochs = max(len(tl) for tl in train_losses)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Vẽ từng fold mờ
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        epochs = range(1, len(tl) + 1)
        ax.plot(epochs, tl, alpha=0.2, color="steelblue")
        ax.plot(epochs, vl, alpha=0.2, color="coral")

    # Vẽ trung bình (pad NaN cho fold ngắn hơn)
    padded_train = np.full((len(train_losses), max_epochs), np.nan)
    padded_val = np.full((len(val_losses), max_epochs), np.nan)
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        padded_train[i, :len(tl)] = tl
        padded_val[i, :len(vl)] = vl

    mean_train = np.nanmean(padded_train, axis=0)
    mean_val = np.nanmean(padded_val, axis=0)
    epochs_range = range(1, max_epochs + 1)
    ax.plot(epochs_range, mean_train, color="steelblue", linewidth=2, label="Train Loss (mean)")
    ax.plot(epochs_range, mean_val, color="coral", linewidth=2, label="Val Loss (mean)")

    ax.set_title("Training & Validation Loss — RatingNet MLP V4", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✅ Loss curves saved: {output_path}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Residual Distribution: Histogram."""
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=80, color="steelblue", alpha=0.7, edgecolor="white")

    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax.axvline(mean_res, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {mean_res:.1f}")
    ax.axvline(mean_res + std_res, color="orange", linestyle=":", linewidth=1.5,
               label=f"±1σ = {std_res:.1f}")
    ax.axvline(mean_res - std_res, color="orange", linestyle=":", linewidth=1.5)

    ax.set_title("Phân phối Sai số (Residuals) — RatingNet MLP V4\n"
                 f"Mean={mean_res:.1f}, Std={std_res:.1f} ELO", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sai số (Predicted - Actual) ELO", fontsize=12)
    ax.set_ylabel("Số lượng ván", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✅ Residual distribution saved: {output_path}")


def plot_ablation_comparison(ablation_results: list, output_path: Path):
    """Bar chart so sánh MAE giữa các ablation configs + XGBoost V3 baseline."""
    names = [r["name"] for r in ablation_results]
    maes = [r["mae_mean"] for r in ablation_results]
    stds = [r["mae_std"] for r in ablation_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#3498db", "#2980b9", "#2471a3", "#1a5276"]  # Xanh đậm dần
    bars = ax.bar(names, maes, yerr=stds, color=colors[:len(names)], alpha=0.85,
                  edgecolor="white", linewidth=1.5, capsize=5)

    # XGBoost V3 baseline (MAE=247.81)
    ax.axhline(y=247.81, color="red", linestyle="--", linewidth=2,
               label="XGBoost V3 (MAE=247.81)")

    # Annotate values
    for bar, mae, std in zip(bars, maes, stds):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 3,
                f'{mae:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title("Ablation Study: MAE theo Feature Config — MLP V4 vs XGBoost V3",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature Config", fontsize=12)
    ax.set_ylabel("MAE (ELO)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  ✅ Ablation comparison saved: {output_path}")


# ── Main ───────────────────────────────────────────────────

def main():
    print(f"{'═' * 60}")
    print("  ĐÁNH GIÁ RATING NET (MLP) V4 — DỰ ĐOÁN ELO LIÊN TỤC")
    print(f"  PyTorch MLP feedforward (256→128→64→1)")
    print(f"{'═' * 60}")

    # ── 1. Load data ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 1: LOAD DATA")
    print(f"{'─' * 60}")
    features_df, feature_cols, X, y = load_data()

    # Chuẩn bị feature subsets cho ablation
    tabular_features = [c for c in feature_cols if c not in ALL_ENGINE]
    tab_a = tabular_features + GROUP_A
    tab_ab = tabular_features + GROUP_A + GROUP_B
    tab_abc = feature_cols  # full

    # ── 2. Full Model ────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 2: FULL MODEL — RatingNet 5-Fold CV")
    print(f"{'─' * 60}")

    full_result = run_mlp_cv(X, y, feature_cols, "D: Full V2 (A+B+C)")

    # ── 3. Ablation Study ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 3: ABLATION STUDY")
    print(f"{'─' * 60}")

    ablation_results = []

    # Config A: Tabular Only
    X_tab = features_df.select(tabular_features).to_numpy().astype(np.float32)
    X_tab = np.nan_to_num(X_tab, nan=0.0)
    res_a = run_mlp_cv(X_tab, y, tabular_features, "A: Tabular Only")
    ablation_results.append(res_a)

    # Config B: Tabular + Nhóm A
    X_tab_a = features_df.select(tab_a).to_numpy().astype(np.float32)
    X_tab_a = np.nan_to_num(X_tab_a, nan=0.0)
    res_b = run_mlp_cv(X_tab_a, y, tab_a, "B: Tabular + Nhóm A")
    ablation_results.append(res_b)

    # Config C: Tabular + Nhóm A + B
    X_tab_ab = features_df.select(tab_ab).to_numpy().astype(np.float32)
    X_tab_ab = np.nan_to_num(X_tab_ab, nan=0.0)
    res_c = run_mlp_cv(X_tab_ab, y, tab_ab, "C: Tabular + A + B")
    ablation_results.append(res_c)

    # Config D đã chạy ở Phase 2
    ablation_results.append(full_result)

    # Bảng tổng hợp
    print(f"\n{'═' * 60}")
    print("  ABLATION SUMMARY — RatingNet MLP V4")
    print(f"{'─' * 60}")
    print(f"  {'Config':<30s} {'#Feat':>6s} {'MAE':>12s} {'RMSE':>12s} {'R²':>12s}")
    print(f"  {'─' * 72}")
    for r in ablation_results:
        print(
            f"  {r['name']:<30s} "
            f"{r['n_features']:>6d} "
            f"{r['mae_mean']:>6.1f}±{r['mae_std']:<4.1f} "
            f"{r['rmse_mean']:>6.1f}±{r['rmse_std']:<4.1f} "
            f"{r['r2_mean']:>6.4f}±{r['r2_std']:<6.4f}"
        )

    # So sánh với XGBoost V3
    print(f"\n  {'─' * 60}")
    print(f"  SO SÁNH VỚI XGBOOST V3 BASELINE:")
    print(f"  XGBoost V3: MAE=247.81, RMSE=315.78, R²=0.6535")
    print(f"  MLP V4    : MAE={full_result['mae_mean']:.2f}, "
          f"RMSE={full_result['rmse_mean']:.2f}, "
          f"R²={full_result['r2_mean']:.4f}")
    delta_mae = full_result['mae_mean'] - 247.81
    print(f"  Delta MAE : {'+' if delta_mae > 0 else ''}{delta_mae:.2f} ELO "
          f"({'tệ hơn' if delta_mae > 0 else 'tốt hơn'})")

    # ── 4. Trực quan hóa ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 4: TRỰC QUAN HÓA")
    print(f"{'─' * 60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_pred_oof = full_result["oof_preds"]

    plot_scatter(y, y_pred_oof, full_result, OUTPUT_DIR / "scatter_actual_vs_predicted.png")
    plot_loss_curves(
        full_result["train_losses"],
        full_result["val_losses"],
        OUTPUT_DIR / "loss_curves.png",
    )
    plot_residuals(y, y_pred_oof, OUTPUT_DIR / "residual_distribution.png")
    plot_ablation_comparison(ablation_results, OUTPUT_DIR / "ablation_comparison.png")

    # ── 5. Lưu kết quả ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PHASE 5: LƯU KẾT QUẢ")
    print(f"{'─' * 60}")

    summary = {
        "model": "RatingNet (MLP)",
        "version": "V4",
        "architecture": "Linear(in,256)→ReLU→Drop(0.3)→Linear(256,128)→ReLU→Drop(0.2)→Linear(128,64)→ReLU→Linear(64,1)",
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
        "training_params": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "patience": PATIENCE,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=5)",
            "loss": "MSELoss",
            "device": str(DEVICE),
        },
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
                "n_features": r["n_features"],
                "mae_mean": r["mae_mean"],
                "mae_std": r["mae_std"],
                "rmse_mean": r["rmse_mean"],
                "rmse_std": r["rmse_std"],
                "r2_mean": r["r2_mean"],
                "r2_std": r["r2_std"],
            }
            for r in ablation_results
        ],
        "xgboost_v3_baseline": {
            "mae_mean": 247.81,
            "rmse_mean": 315.78,
            "r2_mean": 0.6535,
        },
        "comparison": {
            "delta_mae": round(float(delta_mae), 2),
            "verdict": "tốt hơn" if delta_mae < 0 else "tệ hơn",
        },
    }

    results_path = OUTPUT_DIR / "eval_results_rating_net.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Kết quả lưu tại: {results_path}")

    print(f"\n{'═' * 60}")
    print("  HOÀN TẤT — RatingNet MLP V4")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
