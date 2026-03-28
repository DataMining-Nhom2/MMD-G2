"""Pipeline Feature Engineering cho bài toán dự đoán ELO theo ModelBand.

Phiên bản V2 (Stockfish CPL — 11 Features):
- Load dữ liệu sample 30k từ Parquet.
- Transform tabular features (ECO, NumMoves — KHÔNG có GameFormat/BaseTime/Increment).
- Transform engine features bằng Stockfish V2: 11 features theo 3 nhóm A/B/C.

Thay đổi V2 so với V1:
- Nhóm A: Tỷ lệ hóa blunder/mistake/inaccuracy (thay count thô).
- Nhóm B (MỚI): Phân giai đoạn CPL — opening/midgame/endgame.
- Nhóm C (MỚI): WDL probability loss + PV[0] best move match rate.
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
    OPENING_END_MOVE,
    MIDGAME_END_MOVE,
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
# ║              STOCKFISH TRANSFORMER V2                    ║
# ║  (11 Features: Nhóm A / B / C)                          ║
# ╚══════════════════════════════════════════════════════════╝


# Ngưỡng phân loại lỗi (đơn vị: centipawns)
BLUNDER_THRESHOLD = 300
MISTAKE_THRESHOLD = 100
INACCURACY_THRESHOLD = 50


def _default_features() -> dict[str, float]:
    """Trả về dict features mặc định khi ván không phân tích được."""
    return {
        # Nhóm A
        "avg_cpl": 0.0,
        "cpl_std": 0.0,
        "blunder_rate": 0.0,
        "mistake_rate": 0.0,
        "inaccuracy_rate": 0.0,
        # Nhóm B
        "opening_cpl": float("nan"),
        "midgame_cpl": float("nan"),
        "endgame_cpl": float("nan"),
        # Nhóm C
        "avg_wdl_loss": 0.0,
        "max_wdl_loss": 0.0,
        "best_move_match_rate": 0.0,
    }


def _stockfish_worker_chunk(args: tuple[list[str], str, int, int, int]) -> list[dict[str, float]]:
    """Worker chạy trong subprocess: phân tích một chunk ván bằng Stockfish V2."""
    chunk_san, engine_path, depth, threads, hash_mb = args
    import chess.engine

    sf = StockfishTransformer(engine_path, depth, threads, hash_mb)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": threads, "Hash": hash_mb})

    results = []
    try:
        for san in chunk_san:
            try:
                res = sf.analyze_game(san, engine)
            except Exception:
                res = _default_features()
            results.append(res)
    finally:
        engine.quit()
    return results


class StockfishTransformer:
    """Phân tích chất lượng nước đi bằng Stockfish engine — Phiên bản V2.

    Trích xuất 11 features theo 3 nhóm:

    Nhóm A — Tỷ lệ hóa toàn ván:
        avg_cpl, cpl_std,
        blunder_rate, mistake_rate, inaccuracy_rate

    Nhóm B — Phân giai đoạn CPL:
        opening_cpl  (nước 1-10)
        midgame_cpl  (nước 11-30, NaN nếu ván ngắn)
        endgame_cpl  (nước 31+,   NaN nếu ván ngắn)

    Nhóm C — WDL & PV Match:
        avg_wdl_loss        (trung bình mất win-probability)
        max_wdl_loss        (mất nhiều nhất trong 1 nước)
        best_move_match_rate (tỷ lệ đi trùng PV[0] của engine)
    """

    # Tên các cột output V2 (11 features)
    OUTPUT_COLUMNS = [
        # Nhóm A
        "avg_cpl",
        "cpl_std",
        "blunder_rate",
        "mistake_rate",
        "inaccuracy_rate",
        # Nhóm B
        "opening_cpl",
        "midgame_cpl",
        "endgame_cpl",
        # Nhóm C
        "avg_wdl_loss",
        "max_wdl_loss",
        "best_move_match_rate",
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
        """Chuyển PovScore về centipawns từ góc nhìn của bên vừa đi.

        Trả về None nếu là mate score (không tính CPL).
        """
        cp = score.pov(turn).score()
        return float(cp) if cp is not None else None

    @staticmethod
    def _wdl_win_prob(score: chess.engine.PovScore, turn: chess.Color) -> float | None:
        """Lấy xác suất thắng từ WDL của Stockfish NNUE, normalize về [0, 1].

        Trả về None nếu WDL không khả dụng.
        """
        try:
            wdl = score.pov(turn).wdl()
            return wdl.wins / 1000.0
        except Exception:
            return None

    def analyze_game(
        self, moves_san: str, engine: chess.engine.SimpleEngine
    ) -> dict[str, float]:
        """Phân tích 1 ván cờ, trả về dict 11 features V2.

        Quy trình:
        1. Phân tích từng nước: gọi engine.analyse() trước khi đi nước.
        2. Thu thập: CPL, WDL win-prob trước, PV[0] (best move).
        3. So sánh nước thực tế vs PV[0] để tính best_move_match.
        4. Thực hiện nước đi, lấy WDL win-prob sau nước.
        5. Tính WDL loss = win_prob_before - win_prob_after.
        6. Tổng hợp 11 features theo 3 nhóm A/B/C.
        """
        board = chess.Board()
        cpls: list[float] = []          # CPL từng nước (có thể là NaN)
        wdl_losses: list[float] = []    # Win-prob loss từng nước
        best_move_matches: int = 0      # Số nước trùng PV[0]
        valid_moves: int = 0            # Số nước đi hợp lệ được phân tích

        # Parse chuỗi SAN thành danh sách nước đi
        raw_tokens = (moves_san or "").split()
        san_moves: list[str] = [
            t for t in raw_tokens
            if not t.endswith(".") and t not in ("1-0", "0-1", "1/2-1/2", "*")
        ]

        limit = chess.engine.Limit(depth=self.depth)

        for san in san_moves:
            try:
                move = board.parse_san(san)
            except Exception:
                # Nước đi không hợp lệ → bỏ qua
                continue

            turn = board.turn  # Bên sắp đi

            # ── Phân tích TRƯỚC khi đi nước ──────────────────────────
            try:
                info_before = engine.analyse(
                    board, limit, info=chess.engine.INFO_ALL
                )
                prev_cp = self._score_to_cp(info_before["score"], turn)
                prev_win = self._wdl_win_prob(info_before["score"], turn)
                best_move = info_before["pv"][0] if info_before.get("pv") else None
            except Exception:
                prev_cp = None
                prev_win = None
                best_move = None

            # ── So sánh với nước thực tế (Nhóm C: PV Match) ─────────
            if best_move is not None and best_move == move:
                best_move_matches += 1

            # ── Thực hiện nước đi ─────────────────────────────────────
            board.push(move)
            valid_moves += 1

            # ── Phân tích SAU khi đi nước ────────────────────────────
            try:
                info_after = engine.analyse(board, limit, info=chess.engine.INFO_ALL)
                post_cp = self._score_to_cp(info_after["score"], turn)
                post_win = self._wdl_win_prob(info_after["score"], turn)
            except Exception:
                post_cp = None
                post_win = None

            # ── Tính CPL (Nhóm A & B) ────────────────────────────────
            if prev_cp is not None and post_cp is not None:
                cpls.append(max(0.0, prev_cp - post_cp))
            else:
                # Gán NaN để giữ đúng mapping index nước đi → CPL
                cpls.append(float("nan"))

            # ── Tính WDL loss (Nhóm C) ───────────────────────────────
            if prev_win is not None and post_win is not None:
                wdl_losses.append(max(0.0, prev_win - post_win))

        # ── Trường hợp ván không có nước đi hợp lệ ───────────────────
        if valid_moves == 0:
            return _default_features()

        # ── Tổng hợp Nhóm A: Tỷ lệ hóa ──────────────────────────────
        arr = np.array(cpls, dtype=np.float32)          # có thể chứa NaN
        arr_valid = arr[~np.isnan(arr)]                  # chỉ lấy giá trị hợp lệ
        n = float(valid_moves)

        if arr_valid.size > 0:
            avg_cpl = float(np.mean(arr_valid))
            cpl_std = float(np.std(arr_valid))
            blunder_rate = float(np.sum(arr_valid > BLUNDER_THRESHOLD)) / n
            mistake_rate = float(
                np.sum((arr_valid > MISTAKE_THRESHOLD) & (arr_valid <= BLUNDER_THRESHOLD))
            ) / n
            inaccuracy_rate = float(
                np.sum((arr_valid > INACCURACY_THRESHOLD) & (arr_valid <= MISTAKE_THRESHOLD))
            ) / n
        else:
            avg_cpl = cpl_std = blunder_rate = mistake_rate = inaccuracy_rate = 0.0

        # ── Tổng hợp Nhóm B: Phân giai đoạn ─────────────────────────
        opening_slice = arr[:OPENING_END_MOVE]
        opening_valid = opening_slice[~np.isnan(opening_slice)]
        midgame_slice = arr[OPENING_END_MOVE:MIDGAME_END_MOVE]
        midgame_valid = midgame_slice[~np.isnan(midgame_slice)]
        endgame_slice = arr[MIDGAME_END_MOVE:]
        endgame_valid = endgame_slice[~np.isnan(endgame_slice)]

        opening_cpl = float(np.mean(opening_valid)) if opening_valid.size > 0 else float("nan")
        midgame_cpl = float(np.mean(midgame_valid)) if midgame_valid.size > 0 else float("nan")
        endgame_cpl = float(np.mean(endgame_valid)) if endgame_valid.size > 0 else float("nan")

        # ── Tổng hợp Nhóm C: WDL & PV Match ─────────────────────────
        if wdl_losses:
            wdl_arr = np.array(wdl_losses, dtype=np.float32)
            avg_wdl_loss = float(np.mean(wdl_arr))
            max_wdl_loss = float(np.max(wdl_arr))
        else:
            avg_wdl_loss = max_wdl_loss = 0.0

        best_move_match_rate = best_move_matches / n

        return {
            # Nhóm A
            "avg_cpl": avg_cpl,
            "cpl_std": cpl_std,
            "blunder_rate": blunder_rate,
            "mistake_rate": mistake_rate,
            "inaccuracy_rate": inaccuracy_rate,
            # Nhóm B
            "opening_cpl": opening_cpl,
            "midgame_cpl": midgame_cpl,
            "endgame_cpl": endgame_cpl,
            # Nhóm C
            "avg_wdl_loss": avg_wdl_loss,
            "max_wdl_loss": max_wdl_loss,
            "best_move_match_rate": best_move_match_rate,
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

        # Lấy số cores (để lại 2 core cho OS)
        max_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 4

        results: list[dict[str, float]] = []

        print(f"\n  Khởi tạo {max_workers} tiến trình Stockfish chạy song song...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for chunk_res in tqdm(
                executor.map(_stockfish_worker_chunk, args_list),
                total=len(args_list),
                desc="Stockfish Parallel V2",
                unit="chunk",
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
    StockfishTransformer V2 (11 features) để tạo feature matrix cuối cùng.
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

        Kết hợp: Target + Tabular (ECO, NumMoves) + Engine V2 (11 features).
        """
        df = self.ensure_target_column(df)

        if not self.is_fitted:
            self.fit(df)

        # Tabular features (ECO + NumMoves)
        tabular_df = self.tabular.transform(df)

        # Stockfish features V2 (11 features)
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
