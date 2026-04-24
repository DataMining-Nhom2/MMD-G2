import os
import json
import warnings
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

warnings.filterwarnings('ignore')

def main():
    print("═" * 60)
    print("  PHASE 5: ĐÁNH GIÁ XGBOOST BASELINE (STOCKFISH CPL)")
    print("─" * 60)

    # 1. Load Data
    data_path = 'data/features/sample_30k_features.parquet'
    df = pl.read_parquet(data_path)
    print(f"  Load data: {df.shape[0]} ván x {df.shape[1]} features")

    # Target & Features
    y = df['ModelBand'].to_numpy()
    features = [c for c in df.columns if c != 'ModelBand']
    X = df.select(features).to_numpy()

    engine_cols = ['avg_cpl', 'blunder_count', 'mistake_count', 'inaccuracy_count', 'max_cpl', 'cpl_std']
    tabular_features = [c for c in features if c not in engine_cols]
    X_tabular = df.select(tabular_features).to_numpy()

    # 2. Định nghĩa Model (XGBoost)
    params = {
        'objective': 'multi:softmax',
        'num_class': 5,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    }

    def eval_baseline(X_data, name):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, f1s = [], []
        
        for train_idx, val_idx in skf.split(X_data, y):
            X_tr, y_tr = X_data[train_idx], y[train_idx]
            X_va, y_va = X_data[val_idx], y[val_idx]
            
            clf = xgb.XGBClassifier(**params)
            # early stopping không bắt buộc với n_estimators=100 nhưng giúp an toàn
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            preds = clf.predict(X_va)
            accs.append(accuracy_score(y_va, preds))
            f1s.append(f1_score(y_va, preds, average='macro'))
            
        print(f"\n  [{name}]")
        print(f"  * Accuracy : {np.mean(accs)*100:.2f}% (±{np.std(accs)*100:.2f}%)")
        print(f"  * Macro F1 : {np.mean(f1s)*100:.2f}% (±{np.std(f1s)*100:.2f}%)")

    # 3. Ablation Study
    print("\n  Đang train 5-Fold CV (Tabular Only)...")
    eval_baseline(X_tabular, "Tabular Only (ECO, NumMoves)")

    print("\n  Đang train 5-Fold CV (Stockfish + Tabular)...")
    eval_baseline(X, "✅ STOCKFISH CPL + Tabular")

    print(f"\n{'═' * 60}")

    # 4. Generate Jupyter Notebook cho Task 5
    notebook_content = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "# Phase 5: Stockfish CPL Baseline\n",
        "Đây là notebook đánh giá hiệu năng XGBoost trên tập features 113 cột (Stockfish CPL + Tabular)."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "import polars as pl\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import StratifiedKFold\n",
        "from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report\n",
        "import xgboost as xgb\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "print('Libraries loaded.')"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Load data\n",
        "df = pl.read_parquet('../data/features/sample_30k_features.parquet')\n",
        "print(f'Shape: {df.shape}')\n",
        "\n",
        "y = df['ModelBand'].to_numpy()\n",
        "features = [c for c in df.columns if c != 'ModelBand']\n",
        "X = df.select(features).to_numpy()\n",
        "\n",
        "engine_cols = ['avg_cpl', 'blunder_count', 'mistake_count', 'inaccuracy_count', 'max_cpl', 'cpl_std']\n",
        "tabular_features = [c for c in features if c not in engine_cols]\n",
        "X_tabular = df.select(tabular_features).to_numpy()\n"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "def evaluate_model(X_data, y_target, feat_names, title):\n",
        "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "    accs, f1s = [], []\n",
        "    oof_preds = np.zeros_like(y_target)\n",
        "    importances = np.zeros(X_data.shape[1])\n",
        "    \n",
        "    params = {\n",
        "        'objective': 'multi:softmax',\n",
        "        'num_class': 5,\n",
        "        'eval_metric': 'mlogloss',\n",
        "        'max_depth': 6,\n",
        "        'learning_rate': 0.1,\n",
        "        'n_estimators': 100,\n",
        "        'random_state': 42,\n",
        "        'n_jobs': -1\n",
        "    }\n",
        "    \n",
        "    for train_idx, val_idx in skf.split(X_data, y_target):\n",
        "        X_tr, y_tr = X_data[train_idx], y_target[train_idx]\n",
        "        X_va, y_va = X_data[val_idx], y_target[val_idx]\n",
        "        \n",
        "        clf = xgb.XGBClassifier(**params)\n",
        "        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)\n",
        "        \n",
        "        preds = clf.predict(X_va)\n",
        "        oof_preds[val_idx] = preds\n",
        "        \n",
        "        accs.append(accuracy_score(y_va, preds))\n",
        "        f1s.append(f1_score(y_va, preds, average='macro'))\n",
        "        importances += clf.feature_importances_ / skf.n_splits\n",
        "        \n",
        "    print(f'=== {title} ===')\n",
        "    print(f'Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%')\n",
        "    print(f'Macro F1: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%')\n",
        "    \n",
        "    return oof_preds, importances\n",
        "\n",
        "print('TABULAR ONLY (Ablation Study):')\n",
        "preds_tab, imp_tab = evaluate_model(X_tabular, y, tabular_features, 'Tabular Only')\n",
        "\n",
        "print('\\nSTOCKFISH + TABULAR (Full Features):')\n",
        "preds_all, imp_all = evaluate_model(X, y, features, 'Stockfish + Tabular')\n"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "cm = confusion_matrix(y, preds_all)\n",
        "plt.figure(figsize=(8,6))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
        "            xticklabels=['Beginner','Intermediate','Advanced','Expert','Master'],\n",
        "            yticklabels=['Beginner','Intermediate','Advanced','Expert','Master'])\n",
        "plt.title('Confusion Matrix - Stockfish CPL XGBoost')\n",
        "plt.xlabel('Predicted')\n",
        "plt.ylabel('Actual')\n",
        "plt.show()\n"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "top_idx = np.argsort(imp_all)[-15:]\n",
        "plt.figure(figsize=(10,6))\n",
        "plt.barh(np.array(features)[top_idx], imp_all[top_idx])\n",
        "plt.title('Top 15 Feature Importances (Stockfish + Tabular)')\n",
        "plt.show()\n"
       ]
      }
     ],
     "metadata": {},
     "nbformat": 4,
     "nbformat_minor": 4
    }

    out_dir = Path('notebooks')
    out_dir.mkdir(exist_ok=True, parents=True)
    nb_path = out_dir / 'stockfish-baseline.ipynb'
    with nb_path.open('w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    print(f"  Đã tự động tạo Jupyter Notebook: {nb_path}")

if __name__ == '__main__':
    main()
