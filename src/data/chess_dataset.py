"""Chess Dataset — PyTorch Dataset cho pipeline DL Rating Net.

Load dữ liệu từ parquet (sample_30k_dl.parquet), replay ván cờ thành
board state tensors (12×8×8), parse clock time và CPL sequences.

Tương thích với kiến trúc RatingNet (CNN + Bi-LSTM) theo paper arXiv:2409.11506.

Cách dùng:
    from src.data.chess_dataset import ChessDataset, collate_chess_batch
    from torch.utils.data import DataLoader

    dataset = ChessDataset("data/processed/sample_30k_dl.parquet")
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_chess_batch)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.data.board_encoder import replay_game_to_boards


class ChessDataset(Dataset):
    """PyTorch Dataset: load ván cờ → board tensors + clock + metadata.

    Mỗi item trả về dict:
        positions:  Tensor [T, 12, 8, 8]   — Board states
        clocks:     Tensor [T]              — Thời gian còn lại (giây, chuẩn hóa)
        cpls:       Tensor [T]              — CPL per-move (nếu có, else 0)
        blunders:   Tensor [T]              — Blunder flag (CPL > threshold)
        targets:    Tensor [2]              — [WhiteELO, BlackELO] (chuẩn hóa)
        length:     int                     — Số nước đi thực tế
        time_control: str                   — Thể thức
    """

    # Thống kê chuẩn hóa (dùng giá trị từ paper làm default, 
    # có thể tính lại từ dataset)
    RATINGS_MEAN = 1514.0
    RATINGS_STD = 366.0
    CLOCKS_MEAN = 273.0
    CLOCKS_STD = 380.0
    CPL_MEAN = 50.0   # Ước tính, sẽ tính lại sau
    CPL_STD = 100.0

    BLUNDER_THRESHOLD = 200  # CPL > 200 = blunder

    def __init__(
        self,
        parquet_path: str | Path,
        max_moves: int = 150,
        cpl_parquet_path: str | Path | None = None,
    ):
        """
        Args:
            parquet_path: Đường dẫn file parquet chứa ván cờ + clock.
            max_moves: Số nước tối đa (truncation). Default 150 (P96).
            cpl_parquet_path: File parquet chứa CPL sequences (optional).
        """
        self.max_moves = max_moves

        # Load data chính
        self.df = pl.read_parquet(str(parquet_path))
        print(f"  [ChessDataset] Loaded {self.df.height} ván từ {parquet_path}")

        # Load CPL data nếu có
        self.cpl_data = None
        if cpl_parquet_path and Path(cpl_parquet_path).exists():
            self.cpl_data = pl.read_parquet(str(cpl_parquet_path))
            print(f"  [ChessDataset] Loaded CPL data: {self.cpl_data.height} ván")

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, idx: int) -> dict:
        row = self.df.row(idx, named=True)

        # ── 1. Replay ván cờ → Board State Tensors ──
        moves_san = row["Moves"]
        board_arrays = replay_game_to_boards(moves_san)

        if len(board_arrays) == 0:
            # Ván lỗi → trả về 1 board trống
            board_arrays = [np.zeros((12, 8, 8), dtype=np.float32)]

        # Truncate
        board_arrays = board_arrays[:self.max_moves]
        length = len(board_arrays)

        # Stack thành tensor [T, 12, 8, 8]
        positions = torch.tensor(np.stack(board_arrays), dtype=torch.float32)

        # ── 2. Clock Time ──
        clock_seq_raw = json.loads(row.get("ClockSeq", "[]") or "[]")
        clock_seq = []
        for c in clock_seq_raw[:self.max_moves]:
            if c is not None:
                clock_seq.append((c - self.CLOCKS_MEAN) / self.CLOCKS_STD)
            else:
                clock_seq.append(0.0)

        # Pad/truncate to match length
        while len(clock_seq) < length:
            clock_seq.append(0.0)
        clock_seq = clock_seq[:length]
        clocks = torch.tensor(clock_seq, dtype=torch.float32)

        # ── 3. CPL + Blunder ──
        cpls_raw = self._get_cpl_for_game(idx)
        cpl_seq = []
        blunder_seq = []
        for c in cpls_raw[:self.max_moves]:
            if c is not None:
                normalized_cpl = (c - self.CPL_MEAN) / self.CPL_STD
                cpl_seq.append(normalized_cpl)
                blunder_seq.append(1.0 if c > self.BLUNDER_THRESHOLD else 0.0)
            else:
                cpl_seq.append(0.0)
                blunder_seq.append(0.0)

        while len(cpl_seq) < length:
            cpl_seq.append(0.0)
            blunder_seq.append(0.0)
        cpl_seq = cpl_seq[:length]
        blunder_seq = blunder_seq[:length]

        cpls = torch.tensor(cpl_seq, dtype=torch.float32)
        blunders = torch.tensor(blunder_seq, dtype=torch.float32)

        # ── 4. Targets (ELO) ──
        white_elo = float(row["WhiteElo"])
        black_elo = float(row["BlackElo"])
        targets = torch.tensor([white_elo, black_elo], dtype=torch.float32)
        targets = (targets - self.RATINGS_MEAN) / self.RATINGS_STD

        # ── 5. Metadata ──
        time_control = categorize_from_row(row)

        return {
            "positions": positions,     # [T, 12, 8, 8]
            "clocks": clocks,           # [T]
            "cpls": cpls,               # [T]
            "blunders": blunders,        # [T]
            "targets": targets,          # [2]
            "length": length,            # int
            "time_control": time_control,  # str
            "result": row.get("Result", ""),
        }

    def _get_cpl_for_game(self, idx: int) -> list:
        """Lấy CPL sequence cho ván thứ idx. Trả về list rỗng nếu chưa có."""
        if self.cpl_data is not None and idx < self.cpl_data.height:
            cpl_str = self.cpl_data.row(idx, named=True).get("cpl_seq", "[]")
            if cpl_str:
                return json.loads(cpl_str)
        return []


def categorize_from_row(row: dict) -> str:
    """Xác định thể thức từ row data."""
    fmt = row.get("GameFormat", "")
    if fmt:
        return fmt
    # Fallback: tính từ BaseTime + Increment
    try:
        base = int(row.get("BaseTime", 0))
        inc = int(row.get("Increment", 0))
        est = base + 40 * inc
        if est < 29:
            return "UltraBullet"
        elif est < 179:
            return "Bullet"
        elif est < 479:
            return "Blitz"
        elif est < 1499:
            return "Rapid"
        else:
            return "Classical"
    except (ValueError, TypeError):
        return "Unknown"


def collate_chess_batch(batch: list[dict]) -> dict:
    """Collate function cho DataLoader — pad sequences.

    Args:
        batch: List of dicts từ ChessDataset.__getitem__

    Returns:
        Dict of batched tensors, padded to max length in batch.
    """
    positions = pad_sequence(
        [item["positions"] for item in batch], batch_first=True
    )  # [B, T_max, 12, 8, 8]

    clocks = pad_sequence(
        [item["clocks"] for item in batch], batch_first=True
    )  # [B, T_max]

    cpls = pad_sequence(
        [item["cpls"] for item in batch], batch_first=True
    )  # [B, T_max]

    blunders = pad_sequence(
        [item["blunders"] for item in batch], batch_first=True
    )  # [B, T_max]

    targets = torch.stack([item["targets"] for item in batch])  # [B, 2]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.int64)  # [B]
    time_controls = [item["time_control"] for item in batch]
    results = [item["result"] for item in batch]

    return {
        "positions": positions,
        "clocks": clocks,
        "cpls": cpls,
        "blunders": blunders,
        "targets": targets,
        "lengths": lengths,
        "time_controls": time_controls,
        "results": results,
    }
