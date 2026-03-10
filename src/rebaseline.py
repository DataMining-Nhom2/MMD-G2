"""Re-baseline XGBoost cho feature engineering (Phase 5).

Mục tiêu:
- Train/eval trên feature store train/val.
- Báo cáo accuracy + macro F1 + breakdown theo lớp.
- Chạy ablation theo nhóm feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, classification_report, f1_score
import xgboost as xgb

# Đảm bảo import được package src khi chạy script trực tiếp.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_config import FeatureConfig


@dataclass(slots=True)
class RebaselineConfig:
    """Cấu hình train/eval XGBoost cho re-baseline."""

    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.08
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    random_state: int = 42
    tree_method: str = "hist"
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"


class RebaselineRunner:
    """Runner cho train/eval/ablation trên feature store."""

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        model_config: RebaselineConfig | None = None,
    ) -> None:
        self.feature_config = feature_config or FeatureConfig()
        self.model_config = model_config or RebaselineConfig()

    def load_features(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Đọc train/val features từ Parquet."""
        train_path = self.feature_config.train_features_file
        val_path = self.feature_config.val_features_file
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Thiếu feature store: {train_path} hoặc {val_path}. "
                "Hãy chạy Phase 4 trước."
            )
        return pl.read_parquet(train_path), pl.read_parquet(val_path)

    @staticmethod
    def _split_xy(
        df: pl.DataFrame, selected_columns: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Tách X/y, tùy chọn chọn subset feature columns."""
        if "ModelBand" not in df.columns:
            raise ValueError("Thiếu cột ModelBand trong feature store.")

        feature_cols = [c for c in df.columns if c != "ModelBand"]
        if selected_columns is not None:
            feature_cols = [c for c in feature_cols if c in selected_columns]

        x = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df["ModelBand"].cast(pl.Int32).to_numpy()
        return x, y, feature_cols

    def _build_model(self, num_class: int) -> xgb.XGBClassifier:
        """Khởi tạo model XGBClassifier theo config."""
        cfg = self.model_config
        return xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            tree_method=cfg.tree_method,
            objective=cfg.objective,
            eval_metric=cfg.eval_metric,
            num_class=num_class,
            n_jobs=-1,
        )

    def train_and_evaluate(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        selected_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Train model và trả metric evaluation."""
        x_train, y_train, used_features = self._split_xy(
            train_df, selected_columns=selected_columns
        )
        x_val, y_val, _ = self._split_xy(val_df, selected_columns=selected_columns)

        num_class = int(max(y_train.max(), y_val.max()) + 1)
        model = self._build_model(num_class=num_class)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_val)
        acc = float(accuracy_score(y_val, y_pred))
        macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
        report = classification_report(y_val, y_pred, digits=4, zero_division=0)

        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        idx_to_name = {f"f{i}": name for i, name in enumerate(used_features)}
        feature_importance = sorted(
            [
                {
                    "feature": idx_to_name.get(key, key),
                    "gain": float(value[0] if isinstance(value, list) else value),
                }
                for key, value in gain.items()
            ],
            key=lambda item: item["gain"],
            reverse=True,
        )

        return {
            "model": model,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "classification_report": report,
            "feature_importance": feature_importance,
            "used_feature_count": len(used_features),
        }

    @staticmethod
    def _ablation_columns(all_columns: list[str]) -> dict[str, list[str]]:
        """Sinh tập cột cho ablation: tabular-only, tabular+sequence, all."""
        non_target = [c for c in all_columns if c != "ModelBand"]

        move_cols = [
            c
            for c in non_target
            if c.startswith("svd_")
            or c
            in {
                "move_entropy",
                "has_castles_ks",
                "has_castles_qs",
                "check_count_15ply",
                "pawn_push_ratio",
                "first_move_e4",
                "first_move_d4",
                "first_move_Nf3",
                "first_move_c4",
                "first_move_other",
            }
        ]
        tabular_cols = [c for c in non_target if c not in move_cols]

        return {
            "tabular_only": tabular_cols,
            "tabular_plus_sequence": tabular_cols + move_cols,
            "all_features": non_target,
        }

    def run_full_rebaseline(self) -> dict[str, Any]:
        """Chạy full re-baseline + ablation và trả summary."""
        train_df, val_df = self.load_features()

        full_result = self.train_and_evaluate(train_df, val_df)
        ablation_sets = self._ablation_columns(train_df.columns)

        ablation_results: dict[str, dict[str, float | int]] = {}
        for name, cols in ablation_sets.items():
            result = self.train_and_evaluate(train_df, val_df, selected_columns=cols)
            ablation_results[name] = {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "used_feature_count": result["used_feature_count"],
            }

        return {
            "full": {
                "accuracy": full_result["accuracy"],
                "macro_f1": full_result["macro_f1"],
                "used_feature_count": full_result["used_feature_count"],
                "classification_report": full_result["classification_report"],
                "top_feature_importance": full_result["feature_importance"][:20],
            },
            "ablation": ablation_results,
        }

    @staticmethod
    def save_feature_importance_plot(
        top_importance: list[dict[str, Any]],
        output_path: Path,
        top_k: int = 20,
    ) -> None:
        """Vẽ và lưu biểu đồ feature importance theo gain."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        top_items = top_importance[:top_k]
        if not top_items:
            return

        names = [str(item["feature"]) for item in top_items][::-1]
        gains = [float(item["gain"]) for item in top_items][::-1]

        plt.figure(figsize=(12, 8))
        plt.barh(names, gains)
        plt.xlabel("Gain")
        plt.ylabel("Feature")
        plt.title("Top Feature Importance (Gain)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def save_summary_json(summary: dict[str, Any], output_path: Path) -> None:
        """Lưu summary re-baseline ra JSON để làm artifact/review."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Entry point chạy re-baseline từ command line."""
    runner = RebaselineRunner()
    summary = runner.run_full_rebaseline()

    print("=== Re-baseline Summary ===")
    print(f"Accuracy: {summary['full']['accuracy']:.4f}")
    print(f"Macro-F1: {summary['full']['macro_f1']:.4f}")
    print(f"Feature count: {summary['full']['used_feature_count']}")
    print("--- Ablation ---")
    for name, result in summary["ablation"].items():
        print(
            f"{name}: acc={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"features={result['used_feature_count']}"
        )

    # Lưu artifact mặc định cho quick review.
    artifact_dir = Path("data/features/smoke")
    RebaselineRunner.save_feature_importance_plot(
        summary["full"]["top_feature_importance"],
        artifact_dir / "feature_importance_top20.png",
        top_k=20,
    )
    RebaselineRunner.save_summary_json(
        summary, artifact_dir / "rebaseline_summary.json"
    )


if __name__ == "__main__":
    main()
