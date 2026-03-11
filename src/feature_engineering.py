"""Pipeline Feature Engineering cho bài toán dự đoán ELO theo ModelBand.

File này cung cấp khung triển khai theo thiết kế:
- Load dữ liệu theo batch từ Parquet.
- Transform tabular features (ECO, time control, numeric).
- Transform move-sequence features (TF-IDF/SVD + feature thủ công).

Ghi chú:
- Đây là khung bước đầu cho Phase 1 (Task 1.1), các bước fit/transform chi tiết
  sẽ được hoàn thiện ở các task tiếp theo trong planning.
"""

from __future__ import annotations

from typing import Iterator, Sequence
from collections import Counter
import re
import json
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import chess
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_config import (
    FeatureConfig,
    MODEL_BAND_LABEL_TO_ID,
    MODEL_BINS,
    TARGET_SOURCE_COLUMNS,
)


class BatchLoader:
    """Load dữ liệu Parquet theo batch để tránh OOM."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def iter_batches(
        self,
        files: Sequence[str],
        columns: Sequence[str],
    ) -> Iterator[pl.DataFrame]:
        """Sinh từng batch DataFrame từ danh sách file Parquet.

        Hiện tại dùng stream scan và collect toàn bộ theo file để giữ khung đơn giản.
        Ở task tối ưu hiệu năng sẽ thay bằng chiến lược chunk thực sự theo row groups.
        """
        for path in files:
            parquet_file = pq.ParquetFile(path)
            available_columns = [c for c in columns if c in parquet_file.schema.names]

            for record_batch in parquet_file.iter_batches(
                batch_size=self.batch_size,
                columns=available_columns,
            ):
                yield pl.from_arrow(record_batch)


class TabularTransformer:
    """Encode các cột tabular thành numeric features."""

    def __init__(self, eco_top_n: int, realtime_mode: bool = True) -> None:
        self.eco_top_n = eco_top_n
        self.realtime_mode = realtime_mode
        self.eco_vocab: list[str] = []
        self.gf_vocab: list[str] = []
        self.basetime_clip_max: float = 10_800.0
        self.increment_clip_max: float = 180.0

    @staticmethod
    def _safe_quantile(series: pl.Series, quantile: float, fallback: float) -> float:
        """Lấy quantile an toàn, trả fallback nếu không đủ dữ liệu."""
        if series.len() == 0:
            return fallback
        value = series.quantile(quantile)
        if value is None or not np.isfinite(value):
            return fallback
        return float(value)

    @staticmethod
    def _sanitize_feature_name(text: str) -> str:
        """Chuẩn hóa tên category để tạo tên cột ổn định."""
        return text.replace(" ", "_").replace("/", "_")

    def _stable_one_hot(
        self,
        df: pl.DataFrame,
        source_col: str,
        categories: Sequence[str],
        output_prefix: str,
    ) -> pl.DataFrame:
        """One-hot với schema ổn định, luôn trả đủ cột theo categories."""
        dummies = df.select(source_col).to_dummies(columns=[source_col], separator="_")
        source_names = [f"{source_col}_{cat}" for cat in categories]

        exprs: list[pl.Expr] = []
        for source_name, cat in zip(source_names, categories):
            out_name = f"{output_prefix}_{self._sanitize_feature_name(cat)}"
            if source_name in dummies.columns:
                exprs.append(pl.col(source_name).cast(pl.Float32).alias(out_name))
            else:
                exprs.append(pl.lit(0.0).cast(pl.Float32).alias(out_name))
        return dummies.select(exprs)

    def fit(self, df: pl.DataFrame) -> None:
        """Fit vocabulary cho ECO và GameFormat."""
        eco_counts = (
            df.select(pl.col("ECO").cast(pl.Utf8).fill_null("UNK"))
            .group_by("ECO")
            .len()
            .sort("len", descending=True)
            .head(self.eco_top_n)
        )
        self.eco_vocab = eco_counts["ECO"].to_list()

        gf_counts = (
            df.select(pl.col("GameFormat").cast(pl.Utf8).fill_null("Unknown"))
            .group_by("GameFormat")
            .len()
            .sort("len", descending=True)
        )
        self.gf_vocab = gf_counts["GameFormat"].to_list()
        if "Unknown" not in self.gf_vocab:
            self.gf_vocab.append("Unknown")

        numeric_df = df.select(
            [
                pl.col("BaseTime").cast(pl.Float32).fill_null(0.0).alias("BaseTime"),
                pl.col("Increment").cast(pl.Float32).fill_null(0.0).alias("Increment"),
            ]
        )
        self.basetime_clip_max = max(
            1.0,
            self._safe_quantile(
                numeric_df["BaseTime"], quantile=0.99, fallback=10_800.0
            ),
        )
        self.increment_clip_max = max(
            1.0,
            self._safe_quantile(
                numeric_df["Increment"],
                quantile=0.99,
                fallback=180.0,
            ),
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Biến đổi tabular features theo vocabulary đã fit."""
        work_df = df.with_columns(
            [
                pl.col("ECO").cast(pl.Utf8).fill_null("UNK").alias("ECO"),
                pl.col("ECO")
                .cast(pl.Utf8)
                .fill_null("UNK")
                .str.slice(0, 1)
                .alias("EcoCategory"),
                pl.col("GameFormat")
                .cast(pl.Utf8)
                .fill_null("Unknown")
                .alias("GameFormat"),
                pl.col("BaseTime")
                .cast(pl.Float32)
                .fill_null(0.0)
                .clip(0.0, self.basetime_clip_max)
                .alias("BaseTime"),
                pl.col("Increment")
                .cast(pl.Float32)
                .fill_null(0.0)
                .clip(0.0, self.increment_clip_max)
                .alias("Increment"),
                pl.col("NumMoves").cast(pl.Float32).fill_null(0.0).alias("NumMoves"),
            ]
        )

        eco_top_df = self._stable_one_hot(
            work_df,
            source_col="ECO",
            categories=self.eco_vocab,
            output_prefix="eco",
        )
        eco_cat_df = self._stable_one_hot(
            work_df,
            source_col="EcoCategory",
            categories=["A", "B", "C", "D", "E"],
            output_prefix="eco_cat",
        )
        gf_df = self._stable_one_hot(
            work_df,
            source_col="GameFormat",
            categories=self.gf_vocab,
            output_prefix="gf",
        )
        numeric_df = work_df.select(
            [
                pl.col("BaseTime").log1p().cast(pl.Float32).alias("basetime_log"),
                pl.col("Increment").log1p().cast(pl.Float32).alias("increment_log"),
                pl.col("NumMoves")
                .clip(0, 100)
                .truediv(100.0)
                .cast(pl.Float32)
                .alias("num_moves_norm"),
            ]
        )

        frames = [eco_top_df, eco_cat_df, gf_df, numeric_df]

        # EloDiff chỉ dùng cho training/offline mode để tránh leakage trong realtime.
        if not self.realtime_mode and {"WhiteElo", "BlackElo"}.issubset(
            set(df.columns)
        ):
            elo_df = df.select(
                (
                    pl.col("WhiteElo").cast(pl.Float32).fill_null(0.0)
                    - pl.col("BlackElo").cast(pl.Float32).fill_null(0.0)
                )
                .clip(-800.0, 800.0)
                .truediv(800.0)
                .cast(pl.Float32)
                .alias("elo_diff_norm")
            )
            frames.append(elo_df)

        return pl.concat(frames, how="horizontal")


class MoveTransformer:
    """Encode move sequences thành numeric features.

    TODO ở các task Phase 3:
    - Fit TF-IDF + SVD.
    - Tích hợp python-chess để parse SAN ổn định cho các feature board-state.
    """

    def __init__(
        self,
        n_ply: int = 10,
        svd_dim: int = 50,
        tfidf_max_features: int = 500,
        tfidf_ngram_range: tuple[int, int] = (1, 2),
        tfidf_min_df: int | float = 2,
        tfidf_max_df: int | float = 0.95,
    ) -> None:
        self.n_ply = n_ply
        self.svd_dim = svd_dim
        self.bigram_vocab: list[str] = []
        self.tfidf_max_features: int = tfidf_max_features
        self.tfidf_ngram_range: tuple[int, int] = tfidf_ngram_range
        self.tfidf_min_df: int | float = tfidf_min_df
        self.tfidf_max_df: int | float = tfidf_max_df
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.svd_model: TruncatedSVD | None = None
        self._effective_svd_dim: int = 0

    @staticmethod
    def _clean_moves_text(moves_san: str) -> str:
        """Làm sạch chuỗi SAN: bỏ số nước đi, kết quả ván, khoảng trắng dư."""
        text = moves_san or ""
        text = re.sub(r"\d+\.(\.\.)?", " ", text)
        text = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*", " ", text)
        text = text.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Chuẩn hóa token SAN: bỏ annotation noise nhưng giữ thông tin tactical chính."""
        t = token.strip()
        if not t:
            return ""
        # Bỏ annotation dạng !, ?, !!, ?! ở cuối token.
        t = re.sub(r"[!?]+$", "", t)
        return t

    def _tokenize(self, moves_san: str, n_ply: int | None = None) -> list[str]:
        """Tách token SAN và lấy tối đa n_ply token đầu tiên."""
        n = n_ply if n_ply is not None else self.n_ply
        clean = self._clean_moves_text(moves_san)
        if not clean:
            return []
        out: list[str] = []
        for raw in clean.split(" "):
            tok = self._normalize_token(raw)
            if tok:
                out.append(tok)
            if len(out) >= n:
                break
        return out

    @staticmethod
    def _bigrams(tokens: Sequence[str]) -> list[str]:
        """Sinh danh sách bigram theo cửa sổ trượt."""
        if len(tokens) < 2:
            return []
        return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    @staticmethod
    def _entropy(tokens: Sequence[str]) -> float:
        """Tính Shannon entropy của phân phối unigram trong một ván."""
        if not tokens:
            return 0.0
        freq = Counter(tokens)
        total = float(len(tokens))
        entropy = 0.0
        for count in freq.values():
            p = count / total
            entropy -= p * np.log2(p)
        return float(entropy)

    @staticmethod
    def _first_move_group(
        tokens: Sequence[str],
    ) -> tuple[float, float, float, float, float]:
        """Map nước đi đầu tiên vào 5 nhóm one-hot."""
        if not tokens:
            return (0.0, 0.0, 0.0, 0.0, 1.0)
        first = tokens[0]
        if first == "e4":
            return (1.0, 0.0, 0.0, 0.0, 0.0)
        if first == "d4":
            return (0.0, 1.0, 0.0, 0.0, 0.0)
        if first == "Nf3":
            return (0.0, 0.0, 1.0, 0.0, 0.0)
        if first == "c4":
            return (0.0, 0.0, 0.0, 1.0, 0.0)
        return (0.0, 0.0, 0.0, 0.0, 1.0)

    def _board_state_features(
        self, tokens: Sequence[str]
    ) -> tuple[float, float, float, float]:
        """Trích xuất feature board-state bằng python-chess từ token SAN.

        Trả về: (has_castles_ks, has_castles_qs, check_count_15ply, pawn_push_ratio)
        """
        board = chess.Board()
        has_castles_ks = 0.0
        has_castles_qs = 0.0
        check_count = 0
        pawn_moves = 0
        parsed_moves = 0

        for idx, san in enumerate(tokens):
            try:
                move = board.parse_san(san)
            except Exception:
                continue

            piece = board.piece_at(move.from_square)
            if piece is not None and piece.piece_type == chess.PAWN and idx < 10:
                pawn_moves += 1

            board.push(move)
            parsed_moves += 1

            if idx < 20:
                if san in ("O-O", "O-O+"):
                    has_castles_ks = 1.0
                if san in ("O-O-O", "O-O-O+"):
                    has_castles_qs = 1.0

            if idx < 15 and board.is_check():
                check_count += 1

        denom = max(1, min(10, parsed_moves))
        pawn_push_ratio = float(pawn_moves / denom)
        return has_castles_ks, has_castles_qs, float(check_count), pawn_push_ratio

    @staticmethod
    def _token_meta_features(tokens: Sequence[str]) -> tuple[float, float, float]:
        """Feature meta từ SAN token: diversity/capture/check ratios."""
        if not tokens:
            return 0.0, 0.0, 0.0

        total = float(len(tokens))
        unique_ratio = float(len(set(tokens)) / total)
        capture_ratio = float(sum(1 for t in tokens if "x" in t) / total)
        check_ratio = float(sum(1 for t in tokens if ("+" in t or "#" in t)) / total)
        return unique_ratio, capture_ratio, check_ratio

    def fit(self, moves_series: pl.Series, sample_size: int = 500_000) -> None:
        """Fit vectorizer/reducer cho move sequence.

        Triển khai Task 3.5/3.6:
        - Fit TF-IDF bigram trên sample.
        - Fit TruncatedSVD và giữ output shape cố định `svd_dim`.
        """
        sample = moves_series.head(sample_size).to_list()
        counter: Counter[str] = Counter()
        docs: list[str] = []
        for raw in sample:
            tokens = self._tokenize(str(raw), n_ply=self.n_ply)
            counter.update(self._bigrams(tokens))
            docs.append(" ".join(tokens))

        # Giữ top bigrams cho Task 3.4.
        self.bigram_vocab = [ng for ng, _ in counter.most_common(200)]

        # Fit TF-IDF n-gram theo cấu hình tối ưu.
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.tfidf_ngram_range,
            lowercase=False,
            token_pattern=r"(?u)\b[^\s]+\b",
            max_features=self.tfidf_max_features,
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            sublinear_tf=True,
        )
        try:
            tfidf = self.tfidf_vectorizer.fit_transform(docs)
        except ValueError:
            # Batch nhỏ có thể bị prune hết term; fallback để pipeline không vỡ.
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=self.tfidf_ngram_range,
                lowercase=False,
                token_pattern=r"(?u)\b[^\s]+\b",
                max_features=self.tfidf_max_features,
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
            )
            tfidf = self.tfidf_vectorizer.fit_transform(docs)

        n_features = int(tfidf.shape[1])
        if n_features <= 1:
            # Không đủ chiều để SVD có ý nghĩa.
            self.svd_model = None
            self._effective_svd_dim = 0
            return

        self._effective_svd_dim = min(self.svd_dim, max(1, n_features - 1))
        self.svd_model = TruncatedSVD(
            n_components=self._effective_svd_dim,
            random_state=42,
        )
        self.svd_model.fit(tfidf)

    def transform(self, moves_series: pl.Series) -> np.ndarray:
        """Trả về ma trận đặc trưng move sequence.

        Cấu trúc cột:
        - svd_0..svd_{svd_dim-1}
        - move_entropy
        - has_castles_ks, has_castles_qs
        - check_count_15ply
        - pawn_push_ratio
        - first_move_e4, first_move_d4, first_move_Nf3, first_move_c4, first_move_other
        - unique_move_ratio, capture_ratio, check_symbol_ratio
        """
        rows: list[list[float]] = []
        docs: list[str] = []
        token_cache: list[list[str]] = []
        for raw in moves_series.to_list():
            tokens = self._tokenize(str(raw), n_ply=self.n_ply)
            token_cache.append(tokens)
            docs.append(" ".join(tokens))

        # TF-IDF -> SVD (nếu đã fit), sau đó pad/truncate để luôn đúng svd_dim.
        svd_part = np.zeros((len(docs), self.svd_dim), dtype=np.float32)
        if (
            self.tfidf_vectorizer is not None
            and self.svd_model is not None
            and self._effective_svd_dim > 0
        ):
            tfidf = self.tfidf_vectorizer.transform(docs)
            reduced = self.svd_model.transform(tfidf).astype(np.float32)
            usable_dim = min(self._effective_svd_dim, reduced.shape[1], self.svd_dim)
            if usable_dim > 0:
                svd_part[:, :usable_dim] = reduced[:, :usable_dim]

        for idx, tokens in enumerate(token_cache):
            ent = self._entropy(tokens)
            ks, qs, checks, pawn_ratio = self._board_state_features(tokens)
            fm = self._first_move_group(tokens)
            unique_ratio, capture_ratio, check_symbol_ratio = self._token_meta_features(
                tokens
            )

            rows.append(
                [
                    *svd_part[idx].tolist(),
                    ent,
                    ks,
                    qs,
                    checks,
                    pawn_ratio,
                    *fm,
                    unique_ratio,
                    capture_ratio,
                    check_symbol_ratio,
                ]
            )

        return np.asarray(rows, dtype=np.float32)


class FeaturePipeline:
    """Pipeline tổng hợp cho Feature Engineering."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        self.loader = BatchLoader(batch_size=self.config.batch_size)
        self.tabular = TabularTransformer(
            eco_top_n=self.config.eco_top_n,
            realtime_mode=self.config.realtime_mode,
        )
        self.move = MoveTransformer(
            n_ply=self.config.n_ply,
            svd_dim=self.config.svd_dim,
            tfidf_max_features=self.config.tfidf_max_features,
            tfidf_ngram_range=(
                self.config.tfidf_ngram_min,
                self.config.tfidf_ngram_max,
            ),
            tfidf_min_df=self.config.tfidf_min_df,
            tfidf_max_df=self.config.tfidf_max_df,
        )
        self.is_fitted = False

    def validate_columns(self, df: pl.DataFrame) -> None:
        """Kiểm tra cột bắt buộc trước khi transform."""
        missing = [c for c in self.config.required_columns if c not in df.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Thiếu cột bắt buộc: {missing_text}")

        # Ít nhất một nguồn target phải tồn tại.
        if not any(col in df.columns for col in TARGET_SOURCE_COLUMNS):
            raise ValueError(
                "Thiếu cột target nguồn: cần ít nhất một trong ['ModelBand', 'EloAvg']"
            )

    def projection_columns(self) -> list[str]:
        """Danh sách cột cần projection khi đọc Parquet."""
        cols = list(self.config.required_columns)
        for c in TARGET_SOURCE_COLUMNS:
            if c not in cols:
                cols.append(c)
        return cols

    def iter_input_batches(self, files: Sequence[str]) -> Iterator[pl.DataFrame]:
        """Đọc dữ liệu đầu vào theo batch với projection cột."""
        return self.loader.iter_batches(files=files, columns=self.projection_columns())

    def _model_band_from_eloavg(self, elo_expr: pl.Expr) -> pl.Expr:
        """Map EloAvg sang ModelBand id theo bins thiết kế."""
        # Bins: [0,1000), [1000,1400), [1400,1800), [1800,2200), [2200,+inf)
        return (
            pl.when(elo_expr < MODEL_BINS[1])
            .then(pl.lit(0))
            .when(elo_expr < MODEL_BINS[2])
            .then(pl.lit(1))
            .when(elo_expr < MODEL_BINS[3])
            .then(pl.lit(2))
            .when(elo_expr < MODEL_BINS[4])
            .then(pl.lit(3))
            .otherwise(pl.lit(4))
            .cast(pl.Int8)
        )

    def ensure_target_column(self, df: pl.DataFrame) -> pl.DataFrame:
        """Đảm bảo luôn có cột ModelBand (Int8) trước khi fit/transform."""
        if "ModelBand" in df.columns:
            if df.schema.get("ModelBand") == pl.Utf8:
                return df.with_columns(
                    pl.col("ModelBand")
                    .replace(MODEL_BAND_LABEL_TO_ID, default=None)
                    .cast(pl.Int8)
                    .alias("ModelBand")
                )
            return df.with_columns(pl.col("ModelBand").cast(pl.Int8).alias("ModelBand"))

        # Fallback theo planning Task 1.4: encode từ EloAvg
        if "EloAvg" not in df.columns:
            raise ValueError(
                "Không thể tạo ModelBand vì thiếu cả 'ModelBand' và 'EloAvg'."
            )

        return df.with_columns(
            self._model_band_from_eloavg(
                pl.col("EloAvg").cast(pl.Float32).fill_null(0.0)
            ).alias("ModelBand")
        )

    def fit(self, df: pl.DataFrame) -> None:
        """Fit các transformer trên một tập dữ liệu đại diện."""
        df = self.ensure_target_column(df)
        self.validate_columns(df)
        self.tabular.fit(df)
        self.move.fit(df["Moves"], sample_size=self.config.tfidf_sample_size)
        self.is_fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform DataFrame đầu vào thành feature matrix."""
        df = self.ensure_target_column(df)
        self.validate_columns(df)

        if not self.is_fitted:
            self.fit(df)

        tabular_df = self.tabular.transform(df)
        move_np = self.move.transform(df["Moves"])
        move_cols = [f"svd_{idx}" for idx in range(self.config.svd_dim)]
        move_cols.extend(
            [
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
                "unique_move_ratio",
                "capture_ratio",
                "check_symbol_ratio",
            ]
        )
        move_df = pl.DataFrame(move_np, schema=move_cols)

        target_df = df.select(pl.col("ModelBand").cast(pl.Int8))

        out_df = pl.concat([target_df, tabular_df, move_df], how="horizontal")
        float_cols = [c for c in out_df.columns if c != "ModelBand"]
        return out_df.with_columns(
            [
                pl.col("ModelBand").cast(pl.Int8),
                *[pl.col(c).cast(pl.Float32) for c in float_cols],
            ]
        )

    def temporal_split_files(self, files: Sequence[str]) -> tuple[list[str], list[str]]:
        """Tách file train/val theo month trong tên file."""
        train_files: list[str] = []
        val_files: list[str] = []
        for file_path in files:
            name = Path(file_path).name
            if "2025-12" in name:
                train_files.append(file_path)
            elif "2026-01" in name:
                val_files.append(file_path)
        return train_files, val_files

    def _estimate_ply_count(self, moves_text: str) -> int:
        """Ước tính số ply bằng tokenizer hiện tại."""
        return len(self.move._tokenize(moves_text, n_ply=10_000))

    def apply_data_quality_gate(
        self, df: pl.DataFrame, split_name: str
    ) -> tuple[pl.DataFrame, dict[str, float | str]]:
        """Loại bản ghi < min_required_ply và áp quality gate theo tỷ lệ drop."""
        if df.height == 0:
            return df, {
                "split": split_name,
                "input_rows": 0,
                "kept_rows": 0,
                "drop_rows": 0,
                "drop_rate": 0.0,
                "status": "pass",
            }

        marked = df.with_columns(
            pl.col("Moves")
            .cast(pl.Utf8)
            .fill_null("")
            .map_elements(self._estimate_ply_count, return_dtype=pl.Int32)
            .alias("_ply_count")
        )

        filtered = marked.filter(
            pl.col("_ply_count") >= self.config.min_required_ply
        ).drop("_ply_count")
        input_rows = int(df.height)
        kept_rows = int(filtered.height)
        drop_rows = input_rows - kept_rows
        drop_rate = (drop_rows * 100.0) / max(1, input_rows)

        status = "pass"
        if drop_rate > self.config.drop_rate_fail_threshold:
            status = "fail"
        elif drop_rate > self.config.drop_rate_pass_threshold:
            status = "warning"

        stats: dict[str, float | str] = {
            "split": split_name,
            "input_rows": float(input_rows),
            "kept_rows": float(kept_rows),
            "drop_rows": float(drop_rows),
            "drop_rate": float(drop_rate),
            "status": status,
        }

        if status == "fail":
            raise ValueError(
                f"Drop rate {drop_rate:.2f}% ở split {split_name} vượt ngưỡng fail {self.config.drop_rate_fail_threshold}%"
            )

        return filtered, stats

    def process_split_to_parquet(
        self, input_files: Sequence[str], output_file: Path, split_name: str
    ) -> dict[str, float | str]:
        """Xử lý một split theo batch và ghi ra Parquet cuối cùng."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        part_dir = output_file.parent / f"_{output_file.stem}_parts"
        part_dir.mkdir(parents=True, exist_ok=True)

        part_files: list[Path] = []
        aggregate_input = 0.0
        aggregate_kept = 0.0
        aggregate_drop = 0.0

        for idx, batch in enumerate(self.iter_input_batches(input_files), start=1):
            batch = self.ensure_target_column(batch)
            filtered, stats = self.apply_data_quality_gate(batch, split_name=split_name)

            aggregate_input += float(stats["input_rows"])
            aggregate_kept += float(stats["kept_rows"])
            aggregate_drop += float(stats["drop_rows"])

            if filtered.height == 0:
                continue

            feature_batch = self.transform(filtered)
            part_file = part_dir / f"part_{idx:05d}.parquet"
            feature_batch.write_parquet(part_file, compression="zstd")
            part_files.append(part_file)

        if not part_files:
            raise ValueError(
                f"Không có batch hợp lệ để ghi output cho split {split_name}."
            )

        scans = [pl.scan_parquet(str(p)) for p in part_files]
        pl.concat(scans).sink_parquet(str(output_file), compression="zstd")
        shutil.rmtree(part_dir, ignore_errors=True)

        total_drop_rate = (aggregate_drop * 100.0) / max(1.0, aggregate_input)
        status = "pass"
        if total_drop_rate > self.config.drop_rate_fail_threshold:
            status = "fail"
        elif total_drop_rate > self.config.drop_rate_pass_threshold:
            status = "warning"

        if status == "fail":
            raise ValueError(
                f"Drop rate tổng {total_drop_rate:.2f}% ở split {split_name} vượt ngưỡng fail {self.config.drop_rate_fail_threshold}%"
            )

        return {
            "split": split_name,
            "input_rows": aggregate_input,
            "kept_rows": aggregate_kept,
            "drop_rows": aggregate_drop,
            "drop_rate": total_drop_rate,
            "status": status,
            "output_file": str(output_file),
        }

    def save_metadata(self, feature_columns: Sequence[str]) -> None:
        """Lưu metadata cho pipeline theo yêu cầu Phase 4."""
        self.config.feature_columns_file.parent.mkdir(parents=True, exist_ok=True)

        with self.config.feature_columns_file.open("w", encoding="utf-8") as f:
            json.dump(list(feature_columns), f, ensure_ascii=False, indent=2)

        with self.config.tfidf_vocab_file.open("wb") as f:
            pickle.dump(self.move.tfidf_vectorizer, f)

        with self.config.svd_components_file.open("wb") as f:
            pickle.dump(self.move.svd_model, f)

    @staticmethod
    def verify_schema_consistency(train_df: pl.DataFrame, val_df: pl.DataFrame) -> None:
        """Xác minh schema train/val giống nhau."""
        if train_df.columns != val_df.columns:
            raise ValueError("Schema train/val không nhất quán về cột.")
        if train_df.dtypes != val_df.dtypes:
            raise ValueError("Schema train/val không nhất quán về kiểu dữ liệu.")
