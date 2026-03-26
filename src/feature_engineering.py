"""Pipeline Feature Engineering cho bài toán dự đoán ELO theo ModelBand.

Phiên bản mới (Stockfish CPL):
- Load dữ liệu sample 30k từ Parquet.
- Transform tabular features (ECO, NumMoves — KHÔNG có GameFormat/BaseTime/Increment).
- Transform engine features bằng Stockfish CPL (thay thế hoàn toàn TF-IDF/SVD cũ).

Thay đổi so với phiên bản cũ:
- Loại bỏ hoàn toàn: MoveTransformer, TfidfVectorizer, TruncatedSVD
- Loại bỏ hoàn toàn: GameFormat, BaseTime, Increment khỏi TabularTransformer
- Thêm mới: StockfishTransformer (CPL, Blunders, Mistakes, Inaccuracies)
"""

from __future__ import annotations

from typing import Iterator, Sequence
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import chess
import chess.engine
import pyarrow.parquet as pq
from tqdm import tqdm

from src.feature_config import (
    FeatureConfig,
    MODEL_BAND_LABEL_TO_ID,
    MODEL_BINS,
    TARGET_SOURCE_COLUMNS,
)


# ╔══════════════════════════════════════════════════════════╗
# ║                     BATCH LOADER                         ║
# ╚══════════════════════════════════════════════════════════╝


class BatchLoader:
    """Load dữ liệu Parquet theo batch để tránh OOM."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def iter_batches(
        self,
        files: Sequence[str],
        columns: Sequence[str],
    ) -> Iterator[pl.DataFrame]:
        """Sinh từng batch DataFrame từ danh sách file Parquet."""
        for path in files:
            parquet_file = pq.ParquetFile(path)
            available_columns = [c for c in columns if c in parquet_file.schema.names]

            for record_batch in parquet_file.iter_batches(
                batch_size=self.batch_size,
                columns=available_columns,
            ):
                yield pl.from_arrow(record_batch)


# ╔══════════════════════════════════════════════════════════╗
# ║                  TABULAR TRANSFORMER                     ║
# ║  (Đã tinh gọn: Loại bỏ GameFormat, BaseTime, Increment) ║
# ╚══════════════════════════════════════════════════════════╝


class TabularTransformer:
    """Encode các cột tabular thành numeric features.

    Theo Design Decision 4: Loại bỏ hoàn toàn GameFormat/BaseTime/Increment
    để tránh nhiễu do thể thức thời gian. Model chỉ học ánh xạ thuần túy
    giữa chiến thuật (CPL) và trình độ (ELO).
    """

    def __init__(self, eco_top_n: int) -> None:
        self.eco_top_n = eco_top_n
        self.eco_vocab: list[str] = []

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
        """Fit vocabulary cho ECO."""
        eco_counts = (
            df.select(pl.col("ECO").cast(pl.Utf8).fill_null("UNK"))
            .group_by("ECO")
            .len()
            .sort("len", descending=True)
            .head(self.eco_top_n)
        )
        self.eco_vocab = eco_counts["ECO"].to_list()

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
                pl.col("NumMoves").cast(pl.Float32).fill_null(0.0).alias("NumMoves"),
            ]
        )

        # ECO one-hot (top N)
        eco_top_df = self._stable_one_hot(
            work_df,
            source_col="ECO",
            categories=self.eco_vocab,
            output_prefix="eco",
        )
        # EcoCategory one-hot (A–E)
        eco_cat_df = self._stable_one_hot(
            work_df,
            source_col="EcoCategory",
            categories=["A", "B", "C", "D", "E"],
            output_prefix="eco_cat",
        )
        # NumMoves normalize
        numeric_df = work_df.select(
            [
                pl.col("NumMoves")
                .clip(0, 100)
                .truediv(100.0)
                .cast(pl.Float32)
                .alias("num_moves_norm"),
            ]
        )

        return pl.concat([eco_top_df, eco_cat_df, numeric_df], how="horizontal")


# ╔══════════════════════════════════════════════════════════╗
# ║                 STOCKFISH TRANSFORMER                    ║
# ║  (MỚI — thay thế hoàn toàn MoveTransformer cũ)          ║
# ╚══════════════════════════════════════════════════════════╝


# Ngưỡng phân loại lỗi (đơn vị: centipawns)
BLUNDER_THRESHOLD = 300
MISTAKE_THRESHOLD = 100
INACCURACY_THRESHOLD = 50


def _stockfish_worker_chunk(args: tuple[list[str], str, int, int, int]) -> list[dict[str, float]]:
    chunk_san, engine_path, depth, threads, hash_mb = args
    import chess.engine
    
    # Khởi tạo instance dùng để gọi analyze_game
    sf = StockfishTransformer(engine_path, depth, threads, hash_mb)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": threads, "Hash": hash_mb})
    
    results = []
    try:
        for san in chunk_san:
            # Catch lỗi từng ván để tránh sập cả chunk
            try:
                res = sf.analyze_game(san, engine)
            except Exception:
                res = {col: 0.0 for col in sf.OUTPUT_COLUMNS}
            results.append(res)
    finally:
        engine.quit()
    return results


class StockfishTransformer:
    """Phân tích chất lượng nước đi bằng Stockfish engine.

    Thay thế hoàn toàn MoveTransformer (TF-IDF/SVD) cũ.
    Trích xuất features dựa trên Centipawn Loss (CPL):
    - avg_cpl: CPL trung bình toàn ván
    - blunder_count: Số nước mất >300cp
    - mistake_count: Số nước mất 100-300cp
    - inaccuracy_count: Số nước mất 50-100cp
    - max_cpl: CPL tệ nhất
    - cpl_std: Độ lệch chuẩn CPL (tính ổn định)
    """

    # Tên các cột output
    OUTPUT_COLUMNS = [
        "avg_cpl",
        "blunder_count",
        "mistake_count",
        "inaccuracy_count",
        "max_cpl",
        "cpl_std",
    ]

    def __init__(
        self,
        engine_path: str,
        depth: int = 10,
        threads: int = 1,
        hash_mb: int = 128,
    ) -> None:
        self.engine_path = engine_path
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb

    def _create_engine(self) -> chess.engine.SimpleEngine:
        """Khởi tạo engine Stockfish."""
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        return engine

    @staticmethod
    def _score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> float | None:
        """Chuyển PovScore về centipawns từ góc nhìn bên đi.

        Trả về None nếu là mate score (không tính CPL).
        """
        cp = score.pov(turn).score()
        return float(cp) if cp is not None else None

    def analyze_game(
        self, moves_san: str, engine: chess.engine.SimpleEngine
    ) -> dict[str, float]:
        """Phân tích 1 ván cờ, trả về dict features CPL.

        Quy trình:
        1. Parse từng nước SAN bằng python-chess
        2. Trước mỗi nước, gọi engine.analyse() để lấy eval tối ưu
        3. Sau khi đi nước, gọi engine.analyse() lại để lấy eval mới
        4. CPL = |eval_trước - eval_sau| (từ góc nhìn bên vừa đi)
        """
        board = chess.Board()
        cpls: list[float] = []

        # Parse chuỗi SAN thành danh sách nước đi
        raw_tokens = (moves_san or "").split()
        san_moves: list[str] = []
        for token in raw_tokens:
            # Bỏ qua số thứ tự nước (1. 2. 3... hoặc 1... )
            if token.endswith(".") or token in ("1-0", "0-1", "1/2-1/2", "*"):
                continue
            san_moves.append(token)

        limit = chess.engine.Limit(depth=self.depth)
        prev_eval: float | None = None

        for san in san_moves:
            try:
                move = board.parse_san(san)
            except Exception:
                # Nước đi không hợp lệ → bỏ qua
                continue

            turn = board.turn  # Bên sắp đi

            # Eval TRƯỚC khi đi nước
            if prev_eval is None:
                try:
                    info = engine.analyse(board, limit)
                    prev_eval = self._score_to_cp(info["score"], turn)
                except Exception:
                    prev_eval = None

            # Thực hiện nước đi
            board.push(move)

            # Eval SAU khi đi nước (từ góc đối thủ, rồi lật dấu)
            try:
                info = engine.analyse(board, limit)
                # Lấy eval từ góc bên VỪA ĐI (không phải bên đang đi)
                post_eval = self._score_to_cp(info["score"], turn)
            except Exception:
                post_eval = None

            # Tính CPL cho nước này
            if prev_eval is not None and post_eval is not None:
                cpl = max(0.0, prev_eval - post_eval)
                cpls.append(cpl)

            # Cập nhật eval cho nước tiếp theo (từ góc bên đang đi)
            try:
                prev_eval = self._score_to_cp(info["score"], board.turn)
            except Exception:
                prev_eval = None

        # Tổng hợp features
        if not cpls:
            return {
                "avg_cpl": 0.0,
                "blunder_count": 0.0,
                "mistake_count": 0.0,
                "inaccuracy_count": 0.0,
                "max_cpl": 0.0,
                "cpl_std": 0.0,
            }

        arr = np.array(cpls, dtype=np.float32)
        return {
            "avg_cpl": float(np.mean(arr)),
            "blunder_count": float(np.sum(arr > BLUNDER_THRESHOLD)),
            "mistake_count": float(
                np.sum((arr > MISTAKE_THRESHOLD) & (arr <= BLUNDER_THRESHOLD))
            ),
            "inaccuracy_count": float(
                np.sum((arr > INACCURACY_THRESHOLD) & (arr <= MISTAKE_THRESHOLD))
            ),
            "max_cpl": float(np.max(arr)),
            "cpl_std": float(np.std(arr)),
        }

    def transform(self, moves_series: pl.Series) -> pl.DataFrame:
        """Batch transform — phân tích toàn bộ Series Moves bằng Stockfish.

        Sử dụng multiprocessing ProcessPoolExecutor để chạy song song.
        """
        import os
        from concurrent.futures import ProcessPoolExecutor
        
        moves_list = [str(x) for x in moves_series.to_list()]
        
        # Chia chunk size 100 ván/chunk
        chunk_size = 100
        chunks = [moves_list[i:i + chunk_size] for i in range(0, len(moves_list), chunk_size)]
        
        args_list = [
            (chunk, self.engine_path, self.depth, self.threads, self.hash_mb)
            for chunk in chunks
        ]
        
        # Lấy số cores (để lại 1 core cho OS nếu có thể)
        max_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
        
        results: list[dict[str, float]] = []
        
        print(f"\n  Khởi tạo {max_workers} tiến trình Stockfish chạy song song...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # map guarantees order
            for chunk_res in tqdm(
                executor.map(_stockfish_worker_chunk, args_list), 
                total=len(args_list), 
                desc="Stockfish Parallel CPL", 
                unit="chunk"
            ):
                results.extend(chunk_res)
                
        return pl.DataFrame(results, schema={
            col: pl.Float32 for col in self.OUTPUT_COLUMNS
        })


# ╔══════════════════════════════════════════════════════════╗
# ║                   FEATURE PIPELINE                       ║
# ╚══════════════════════════════════════════════════════════╝


class FeaturePipeline:
    """Pipeline tổng hợp cho Feature Engineering.

    Kết hợp TabularTransformer (ECO + NumMoves) với
    StockfishTransformer (CPL) để tạo feature matrix cuối cùng.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        self.loader = BatchLoader(batch_size=self.config.batch_size)
        self.tabular = TabularTransformer(eco_top_n=self.config.eco_top_n)
        self.stockfish = StockfishTransformer(
            engine_path=self.config.stockfish_path,
            depth=self.config.stockfish_depth,
            threads=self.config.stockfish_threads,
            hash_mb=self.config.stockfish_hash_mb,
        )
        self.is_fitted = False

    def projection_columns(self) -> list[str]:
        """Danh sách cột cần đọc từ Parquet."""
        cols = list(self.config.required_columns)
        for c in TARGET_SOURCE_COLUMNS:
            if c not in cols:
                cols.append(c)
        return cols

    def _model_band_from_eloavg(self, elo_expr: pl.Expr) -> pl.Expr:
        """Map EloAvg sang ModelBand id theo bins thiết kế."""
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
        """Đảm bảo luôn có cột ModelBand (Int8)."""
        if "ModelBand" in df.columns:
            if df.schema.get("ModelBand") == pl.Utf8:
                return df.with_columns(
                    pl.col("ModelBand")
                    .replace(MODEL_BAND_LABEL_TO_ID, default=None)
                    .cast(pl.Int8)
                    .alias("ModelBand")
                )
            return df.with_columns(pl.col("ModelBand").cast(pl.Int8).alias("ModelBand"))

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
        """Fit TabularTransformer trên dữ liệu đại diện.

        StockfishTransformer không cần fit (stateless).
        """
        df = self.ensure_target_column(df)
        self.tabular.fit(df)
        self.is_fitted = True

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform DataFrame đầu vào thành feature matrix.

        Kết hợp: Target + Tabular (ECO, NumMoves) + Engine (CPL, Blunders).
        """
        df = self.ensure_target_column(df)

        if not self.is_fitted:
            self.fit(df)

        # Tabular features (ECO + NumMoves)
        tabular_df = self.tabular.transform(df)

        # Stockfish features (CPL)
        engine_df = self.stockfish.transform(df["Moves"])

        # Target
        target_df = df.select(pl.col("ModelBand").cast(pl.Int8))

        # Concat tất cả
        out_df = pl.concat([target_df, tabular_df, engine_df], how="horizontal")

        # Đảm bảo mọi feature là Float32
        float_cols = [c for c in out_df.columns if c != "ModelBand"]
        return out_df.with_columns(
            [
                pl.col("ModelBand").cast(pl.Int8),
                *[pl.col(c).cast(pl.Float32) for c in float_cols],
            ]
        )

    def save_metadata(self, feature_columns: Sequence[str]) -> None:
        """Lưu metadata cho pipeline."""
        self.config.feature_columns_file.parent.mkdir(parents=True, exist_ok=True)

        with self.config.feature_columns_file.open("w", encoding="utf-8") as f:
            json.dump(list(feature_columns), f, ensure_ascii=False, indent=2)

        # Lưu cấu hình Stockfish
        stockfish_config = {
            "engine_path": self.config.stockfish_path,
            "depth": self.config.stockfish_depth,
            "threads": self.config.stockfish_threads,
            "hash_mb": self.config.stockfish_hash_mb,
        }
        stockfish_config_file = self.config.feature_columns_file.parent / "stockfish_config.json"
        with stockfish_config_file.open("w", encoding="utf-8") as f:
            json.dump(stockfish_config, f, ensure_ascii=False, indent=2)
