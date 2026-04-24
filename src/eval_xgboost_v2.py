"""Đánh giá XGBoost Baseline V2 (11 features Stockfish).

Theo Planning V2 Task 4.1–4.4:
- Stratified 5-Fold CV trên features V2
- So sánh với V1 (44.24%)
- Ablation Study: (A) Tabular-only, (B) Tabular+Nhóm A, (C) +Nhóm B, (D) Full
- Feature Importance plot
"""

import os
import json
import warnings
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend cho SSH
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Cấu hình ───────────────────────────────────────────────
V2_DATA = "data/features/sample_30k_features_v2.parquet"
V1_ACCURACY = 44.24  # Baseline V1 để so sánh
OUTPUT_DIR = Path("data/results/v2")

# Nhóm features engine V2
GROUP_A = ["avg_cpl", "cpl_std", "blunder_rate", "mistake_rate", "inaccuracy_rate"]
GROUP_B = ["opening_cpl", "midgame_cpl", "endgame_cpl"]
GROUP_C = ["avg_wdl_loss", "max_wdl_loss", "best_move_match_rate"]
ALL_ENGINE = GROUP_A + GROUP_B + GROUP_C

# Tham số XGBoost
XGB_PARAMS = {
    "objective": "multi:softmax",
    "num_class": 5,
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
}


def evaluate_cv(X: np.ndarray, y: np.ndarray, feat_names: list[str], name: str) -> dict:
    """Chạy Stratified 5-Fold CV, trả về metrics + feature importance."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    oof_preds = np.zeros_like(y)
    importances = np.zeros(X.shape[1])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds = clf.predict(X_va)
        oof_preds[val_idx] = preds

        acc = accuracy_score(y_va, preds)
        f1 = f1_score(y_va, preds, average="macro")
        accs.append(acc)
        f1s.append(f1)
        importances += clf.feature_importances_ / skf.n_splits

    mean_acc = np.mean(accs) * 100
    std_acc = np.std(accs) * 100
    mean_f1 = np.mean(f1s) * 100
    std_f1 = np.std(f1s) * 100

    print(f"\n  [{name}]")
    print(f"  * Accuracy : {mean_acc:.2f}% (±{std_acc:.2f}%)")
    print(f"  * Macro F1 : {mean_f1:.2f}% (±{std_f1:.2f}%)")

    return {
        "name": name,
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "f1_mean": mean_f1,
        "f1_std": std_f1,
        "oof_preds": oof_preds,
        "importances": importances,
        "feat_names": feat_names,
    }


def main():
    print(f"{'═' * 60}")
    print("  ĐÁNH GIÁ XGBOOST BASELINE V2 (11 ENGINE FEATURES)")
    print(f"{'─' * 60}")

    # ── 1. Load data V2 ───────────────────────────────────────
    df = pl.read_parquet(V2_DATA)
    print(f"  Load data: {df.shape[0]} ván x {df.shape[1]} cột")

    y = df["ModelBand"].to_numpy()
    all_features = [c for c in df.columns if c != "ModelBand"]
    X_all = df.select(all_features).to_numpy()

    # Phân chia features cho ablation study
    tabular_features = [c for c in all_features if c not in ALL_ENGINE]
    tab_a = tabular_features + GROUP_A
    tab_ab = tabular_features + GROUP_A + GROUP_B
    tab_abc = tabular_features + ALL_ENGINE  # = all_features

    X_tab = df.select(tabular_features).to_numpy()
    X_tab_a = df.select(tab_a).to_numpy()
    X_tab_ab = df.select(tab_ab).to_numpy()

    # ── 2. Ablation Study ─────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  ABLATION STUDY")
    print(f"{'─' * 60}")

    results = []

    print("\n  [A] Tabular Only...")
    results.append(evaluate_cv(X_tab, y, tabular_features, "A: Tabular Only"))

    print("\n  [B] Tabular + Nhóm A (CPL tỷ lệ hóa)...")
    results.append(evaluate_cv(X_tab_a, y, tab_a, "B: Tabular + Nhóm A"))

    print("\n  [C] Tabular + Nhóm A + Nhóm B (Phase CPL)...")
    results.append(evaluate_cv(X_tab_ab, y, tab_ab, "C: Tabular + A + B"))

    print("\n  [D] Full V2 (Tabular + A + B + C)...")
    results.append(evaluate_cv(X_all, y, tab_abc, "D: Full V2 (A+B+C)"))

    # ── 3. So sánh V1 vs V2 ──────────────────────────────────
    full_v2 = results[-1]
    delta = full_v2["accuracy_mean"] - V1_ACCURACY
    print(f"\n{'═' * 60}")
    print(f"  SO SÁNH V1 vs V2")
    print(f"  V1 Accuracy: {V1_ACCURACY:.2f}%")
    print(f"  V2 Accuracy: {full_v2['accuracy_mean']:.2f}%")
    print(f"  Delta      : {'+' if delta > 0 else ''}{delta:.2f}%")
    print(f"{'═' * 60}")

    # ── 4. Ablation Summary Table ─────────────────────────────
    print(f"\n  {'Config':<30s} {'Accuracy':>12s} {'Macro F1':>12s}")
    print(f"  {'─' * 54}")
    for r in results:
        print(
            f"  {r['name']:<30s} "
            f"{r['accuracy_mean']:>5.2f}% ±{r['accuracy_std']:.2f}% "
            f"{r['f1_mean']:>5.2f}% ±{r['f1_std']:.2f}%"
        )

    # ── 5. Feature Importance Plot ────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    imp = full_v2["importances"]
    feat = np.array(full_v2["feat_names"])
    top_idx = np.argsort(imp)[-20:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feat[top_idx], imp[top_idx], color="steelblue")
    ax.set_title("Top 20 Feature Importances — V2 Full (A+B+C)", fontsize=14)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "feature_importance_v2.png"), dpi=150)
    print(f"\n  Feature Importance plot saved: {OUTPUT_DIR / 'feature_importance_v2.png'}")

    # ── 6. Confusion Matrix ───────────────────────────────────
    labels = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
    cm = confusion_matrix(y, full_v2["oof_preds"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix — V2 Full (A+B+C)", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "confusion_matrix_v2.png"), dpi=150)
    print(f"  Confusion Matrix saved: {OUTPUT_DIR / 'confusion_matrix_v2.png'}")

    # ── 7. Classification Report ──────────────────────────────
    report = classification_report(y, full_v2["oof_preds"], target_names=labels)
    print(f"\n  Classification Report (V2 Full):\n{report}")

    # ── 8. Lưu kết quả JSON ──────────────────────────────────
    summary = {
        "v1_accuracy": V1_ACCURACY,
        "ablation": [
            {
                "config": r["name"],
                "accuracy_mean": round(r["accuracy_mean"], 2),
                "accuracy_std": round(r["accuracy_std"], 2),
                "f1_mean": round(r["f1_mean"], 2),
                "f1_std": round(r["f1_std"], 2),
            }
            for r in results
        ],
    }
    with (OUTPUT_DIR / "eval_results_v2.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Results JSON saved: {OUTPUT_DIR / 'eval_results_v2.json'}")

    print(f"\n{'═' * 60}")
    print("  ✅ HOÀN THÀNH ĐÁNH GIÁ V2!")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
