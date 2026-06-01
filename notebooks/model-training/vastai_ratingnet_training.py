# %% [markdown]
# # ♟️ Chess ELO Prediction — CNN-BiLSTM + CPL/Blunder
# 
# **Phiên bản tối ưu cho Vast.ai / Colab Pro**
# - Tự động tải data từ Hugging Face
# - Tự phát hiện số CPU cores → điều chỉnh num_workers
# - Batch size nâng lên 128 (tận dụng GPU mạnh)

# %% [markdown]
# ## 0. Cài đặt & Cấu hình

# %%
# Cài đặt thư viện (chạy 1 lần khi mới tạo máy)
# !pip install -q python-chess datasets pyarrow torch

# %%
# =============================================
# 🔧 CẤU HÌNH — SỬA Ở ĐÂY TRƯỚC KHI CHẠY
# =============================================

import os

CONFIG = {
    # --- Nguồn dữ liệu (Hugging Face) ---
    "hf_dataset": "Khai171/Chess_600k",
    "hf_filename": "sample_600k_dl.parquet",

    # --- Feature flags (bật/tắt để ablation) ---
    "use_clock": True,
    "use_cpl": True,
    "use_blunder": True,

    # --- Model (giữ nguyên paper) ---
    "conv_filters": 32,
    "lstm_layers": 3,
    "lstm_hidden": 64,
    "fc1_hidden": 32,
    "dropout_rate": 0.5,
    "bidirectional": True,
    "max_moves": 150,

    # --- Training (TỐI ƯU CHO VAST.AI) ---
    "batch_size": 128,        # ⬆ Kaggle: 32 → Vast: 128
    "val_batch_size": 512,    # ⬆ Kaggle: 256 → Vast: 512
    "epochs": 60,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "patience": 5,
    "lr_factor": 0.5,
    "num_workers": 14,        # ⬆ Kaggle: 2 → Vast: 14 (tự động điều chỉnh bên dưới)

    # --- Normalization (từ paper) ---
    "ratings_mean": 1514.0,
    "ratings_std": 366.0,
    "clocks_mean": 273.0,
    "clocks_std": 380.0,
    "cpl_mean": 50.0,
    "cpl_std": 100.0,
    "blunder_threshold": 200,
}

# Tự động điều chỉnh num_workers theo CPU khả dụng
available_cpus = os.cpu_count() or 4
# Dùng toàn bộ sức mạnh CPU (chừa lại 2 nhân cho hệ điều hành)
CONFIG["num_workers"] = max(2, available_cpus - 2)
print(f"🖥️  CPU cores: {available_cpus} → num_workers = {CONFIG['num_workers']}")

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import chess
import json
import time

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# %% [markdown]
# ## 1. Tải dữ liệu từ Hugging Face
# 
# Thay vì hardcode đường dẫn Kaggle, tải trực tiếp từ HF Hub.

# %%
from huggingface_hub import hf_hub_download

print("📂 Đang tải dữ liệu từ Hugging Face...")
local_path = hf_hub_download(
    repo_id=CONFIG["hf_dataset"],
    filename=CONFIG["hf_filename"],
    repo_type="dataset",
)
print(f"   ✅ Đã tải về: {local_path}")

df = pd.read_parquet(local_path)
print(f"   Tổng: {len(df)} ván | Columns: {list(df.columns)}")

# %% [markdown]
# ## 2. Board Encoding
# 
# Mỗi vị trí bàn cờ → tensor `[12, 8, 8]` (12 bitboard planes).

# %%
PIECE_TYPE_TO_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

def encode_board(board: chess.Board) -> np.ndarray:
    """Mã hóa bàn cờ thành 12 binary planes [12, 8, 8]."""
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        plane_idx = PIECE_TYPE_TO_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane_idx += 6
        row, col = square // 8, square % 8
        planes[plane_idx, row, col] = 1.0
    return planes

def replay_game_to_boards(moves_san: str, max_moves: int = 150) -> list:
    """Replay ván cờ từ SAN string → list of board tensors."""
    board = chess.Board()
    boards = []
    tokens = moves_san.split()
    tokens = [t for t in tokens if not (t.endswith('.') or t in ('1-0', '0-1', '1/2-1/2', '*'))]
    for token in tokens[:max_moves]:
        try:
            board.push_san(token)
            boards.append(encode_board(board))
        except Exception:
            break
    return boards

# %% [markdown]
# ## 3. Dataset & Collate

# %%
class ChessGamesDataset(Dataset):
    """Dataset cho Chess ELO Prediction — tương thích cả paper gốc lẫn cải tiến."""

    def __init__(self, df, max_moves=150, config=None):
        self.df = df.reset_index(drop=True)
        self.max_moves = max_moves
        self.config = config or CONFIG
        self.has_clock = "ClockSeq" in self.df.columns
        self.has_cpl = "cpl_seq" in self.df.columns
        self.has_time_control = "TimeControl" in self.df.columns

        if not self.has_clock:
            print("  ⚠ Không tìm thấy cột ClockSeq → clock = 0")
        if not self.has_cpl:
            print("  ⚠ Không tìm thấy cột cpl_seq → cpl/blunder = 0")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _safe_parse_json(raw_str):
        """Parse JSON an toàn, thay NaN/Infinity thành null."""
        if pd.isna(raw_str):
            return []
        s = str(raw_str)
        s = s.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _is_valid_number(val):
        """Kiểm tra giá trị hợp lệ (loại bỏ None, NaN, Inf)."""
        if val is None:
            return False
        try:
            f = float(val)
            return not (f != f or f == float('inf') or f == float('-inf'))
        except (ValueError, TypeError):
            return False

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── 1. Board States ──
        board_arrays = replay_game_to_boards(row["Moves"], self.max_moves)
        if not board_arrays:
            board_arrays = [np.zeros((12, 8, 8), dtype=np.float32)]
        length = len(board_arrays)
        positions = torch.tensor(np.stack(board_arrays), dtype=torch.float32)

        # ── 2. Clock Time ──
        clocks = torch.zeros(length, dtype=torch.float32)
        if self.has_clock and self.config["use_clock"]:
            raw = self._safe_parse_json(row["ClockSeq"])
            for i, c in enumerate(raw[:length]):
                if self._is_valid_number(c):
                    clocks[i] = (float(c) - self.config["clocks_mean"]) / self.config["clocks_std"]

        # ── 3. CPL + Blunder (Dùng _safe_parse để chống NaN) ──
        cpls = torch.zeros(length, dtype=torch.float32)
        blunders = torch.zeros(length, dtype=torch.float32)
        if self.has_cpl:
            raw = self._safe_parse_json(row["cpl_seq"])
            for i, c in enumerate(raw[:length]):
                if self._is_valid_number(c):
                    cpl_val = max(0.0, min(float(c), 2000.0))  # Clamp [0, 2000]
                    if self.config["use_cpl"]:
                        cpls[i] = (cpl_val - self.config["cpl_mean"]) / self.config["cpl_std"]
                    if self.config["use_blunder"]:
                        blunders[i] = 1.0 if cpl_val > self.config["blunder_threshold"] else 0.0

        # ── 4. Targets (ELO chuẩn hóa) ──
        white_elo = float(row["WhiteElo"])
        black_elo = float(row["BlackElo"])
        targets = torch.tensor([white_elo, black_elo], dtype=torch.float32)
        targets = (targets - self.config["ratings_mean"]) / self.config["ratings_std"]

        # ── 5. Time Control ──
        time_control = "unknown"
        if self.has_time_control:
            try:
                parts = str(row["TimeControl"]).split("+")
                initial = int(parts[0])
                increment = int(parts[1]) if len(parts) > 1 else 0
                est = initial + 40 * increment
                if est < 29: time_control = "ultrabullet"
                elif est < 179: time_control = "bullet"
                elif est < 479: time_control = "blitz"
                elif est < 1499: time_control = "rapid"
                else: time_control = "classical"
            except (ValueError, IndexError):
                pass

        return {
            'positions': positions,
            'clocks': clocks,
            'cpls': cpls,
            'blunders': blunders,
            'targets': targets,
            'length': length,
            'time_control': time_control,
        }


def collate_fn(batch):
    """Pad sequences trong batch về cùng độ dài."""
    positions = pad_sequence([item['positions'] for item in batch], batch_first=True)
    clocks = pad_sequence([item['clocks'] for item in batch], batch_first=True)
    cpls = pad_sequence([item['cpls'] for item in batch], batch_first=True)
    blunders = pad_sequence([item['blunders'] for item in batch], batch_first=True)
    targets = torch.stack([item['targets'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    time_controls = [item['time_control'] for item in batch]
    return {
        'positions': positions,
        'clocks': clocks,
        'cpls': cpls,
        'blunders': blunders,
        'targets': targets,
        'lengths': lengths,
        'time_controls': time_controls,
    }

# %% [markdown]
# ## 4. Model Architecture — RatingNet (Cải tiến)

# %%
class ChessEloPredictor(nn.Module):
    """RatingNet — CNN + Bi-LSTM, cải tiến với CPL/Blunder."""

    def __init__(self, config=None):
        super().__init__()
        cfg = config or CONFIG
        cf = cfg["conv_filters"]
        self.use_cpl = cfg["use_cpl"]
        self.use_blunder = cfg["use_blunder"]

        # ── CNN Block: 4 tầng (giống hệt paper) ──
        self.conv1 = nn.Conv2d(12, cf, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(cf)
        self.conv2 = nn.Conv2d(cf, cf * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cf * 2)
        self.conv3 = nn.Conv2d(cf * 2, cf * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(cf * 4)
        self.conv4 = nn.Conv2d(cf * 4, cf * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(cf * 8)
        self.pool = nn.AvgPool2d(2, 2)
        self.dropout = nn.Dropout(cfg["dropout_rate"])

        # ── LSTM ──
        extra = 1  # clock (luôn có)
        if self.use_cpl:
            extra += 1
        if self.use_blunder:
            extra += 1

        self.lstm = nn.LSTM(
            input_size=cf * 8 + extra,
            hidden_size=cfg["lstm_hidden"],
            num_layers=cfg["lstm_layers"],
            batch_first=True,
            bidirectional=cfg["bidirectional"],
        )

        # ── FC Head ──
        fc_in = cfg["lstm_hidden"] * 2 if cfg["bidirectional"] else cfg["lstm_hidden"]
        self.fc1 = nn.Linear(fc_in, cfg["fc1_hidden"])
        self.fc2 = nn.Linear(cfg["fc1_hidden"], 2)

    def forward(self, positions, clocks, cpls, blunders, lengths):
        B, T = positions.size(0), positions.size(1)

        # CNN: [B*T, 12, 8, 8] → [B*T, cf*8, 1, 1]
        x = positions.view(-1, 12, 8, 8)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        # Flatten: [B, T, cf*8]
        x = x.view(B, T, -1)

        # Concat extra features
        parts = [x, clocks.unsqueeze(2)]
        if self.use_cpl:
            parts.append(cpls.unsqueeze(2))
        if self.use_blunder:
            parts.append(blunders.unsqueeze(2))
        lstm_input = torch.cat(parts, dim=2)

        # BiLSTM
        packed = pack_padded_sequence(
            lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # FC trên tất cả time-steps
        y = F.leaky_relu(self.fc1(lstm_out))
        y = self.dropout(y)
        all_outputs = self.fc2(y)

        # Lấy output ở nước cuối cùng
        idx = torch.arange(B, device=positions.device)
        last_output = all_outputs[idx, lengths - 1, :]

        return all_outputs, last_output

# %% [markdown]
# ## 5. Training & Evaluation

# %%
def train_one_epoch(model, loader, optimizer, criterion, device, config):
    """Train 1 epoch, trả về average MAE (trên ELO gốc)."""
    model.train()
    total_loss = 0.0
    rm, rs = config["ratings_mean"], config["ratings_std"]

    for batch in loader:
        pos = batch['positions'].to(device)
        clk = batch['clocks'].to(device)
        cpl = batch['cpls'].to(device)
        bld = batch['blunders'].to(device)
        tgt = batch['targets'].to(device)
        lengths = batch['lengths']

        optimizer.zero_grad()
        _, last_out = model(pos, clk, cpl, bld, lengths)

        pred_elo = last_out * rs + rm
        true_elo = tgt * rs + rm
        loss = criterion(pred_elo, true_elo)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, config):
    """Validate, trả về average MAE."""
    model.eval()
    total_loss = 0.0
    rm, rs = config["ratings_mean"], config["ratings_std"]

    for batch in loader:
        pos = batch['positions'].to(device)
        clk = batch['clocks'].to(device)
        cpl = batch['cpls'].to(device)
        bld = batch['blunders'].to(device)
        tgt = batch['targets'].to(device)
        lengths = batch['lengths']

        _, last_out = model(pos, clk, cpl, bld, lengths)
        pred_elo = last_out * rs + rm
        true_elo = tgt * rs + rm
        loss = criterion(pred_elo, true_elo)
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def test_with_breakdown(model, loader, criterion, device, config):
    """Test + MAE breakdown theo thể thức."""
    model.eval()
    total_loss = 0.0
    rm, rs = config["ratings_mean"], config["ratings_std"]
    loss_by_tc = {}
    count_by_tc = {}

    for batch in loader:
        pos = batch['positions'].to(device)
        clk = batch['clocks'].to(device)
        cpl = batch['cpls'].to(device)
        bld = batch['blunders'].to(device)
        tgt = batch['targets'].to(device)
        lengths = batch['lengths']
        tcs = batch['time_controls']

        _, last_out = model(pos, clk, cpl, bld, lengths)
        pred_elo = last_out * rs + rm
        true_elo = tgt * rs + rm
        loss = criterion(pred_elo, true_elo)
        total_loss += loss.item()

        mae_per_item = torch.abs(pred_elo - true_elo).mean(dim=1)
        for i, tc in enumerate(tcs):
            loss_by_tc[tc] = loss_by_tc.get(tc, 0.0) + mae_per_item[i].item()
            count_by_tc[tc] = count_by_tc.get(tc, 0) + 1

    for tc in loss_by_tc:
        if count_by_tc[tc] > 0:
            loss_by_tc[tc] /= count_by_tc[tc]

    return total_loss / len(loader), loss_by_tc

# %% [markdown]
# ## 6. Chạy Training

# %%
def main(config=None):
    cfg = config or CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 60}")
    print(f"  ♟️  RatingNet Training (Vast.ai Edition)")
    print(f"{'─' * 60}")
    print(f"  Device      : {device}")
    print(f"  Data        : {cfg['hf_dataset']}")
    print(f"  Clock       : {'✓' if cfg['use_clock'] else '✗'}")
    print(f"  CPL         : {'✓' if cfg['use_cpl'] else '✗ (ablation)'}")
    print(f"  Blunder     : {'✓' if cfg['use_blunder'] else '✗ (ablation)'}")
    print(f"  BiLSTM      : {'✓' if cfg['bidirectional'] else '✗'}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  Num workers : {cfg['num_workers']}")
    print(f"  Epochs      : {cfg['epochs']}")
    print(f"  LR          : {cfg['lr']}")
    print(f"{'═' * 60}\n")

    # ── Split: 70/20/10 (random_state=42 để tái lập) ──
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

    train_ds = ChessGamesDataset(train_df, max_moves=cfg["max_moves"], config=cfg)
    val_ds = ChessGamesDataset(val_df, max_moves=cfg["max_moves"], config=cfg)
    test_ds = ChessGamesDataset(test_df, max_moves=cfg["max_moves"], config=cfg)
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # persistent_workers=True: Giữ worker processes sống, không tạo lại mỗi epoch
    pw = cfg["num_workers"] > 0
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=cfg["num_workers"],
                              pin_memory=True, persistent_workers=pw)
    val_loader = DataLoader(val_ds, batch_size=cfg["val_batch_size"], shuffle=False,
                            collate_fn=collate_fn, num_workers=cfg["num_workers"],
                            pin_memory=True, persistent_workers=pw)
    test_loader = DataLoader(test_ds, batch_size=cfg["val_batch_size"], shuffle=False,
                             collate_fn=collate_fn, num_workers=cfg["num_workers"],
                             pin_memory=True, persistent_workers=pw)

    # ── Model ──
    model = ChessEloPredictor(config=cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {total_params:,} parameters")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  patience=cfg["patience"],
                                  factor=cfg["lr_factor"])

    # ── Model save dir ──
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)

    # ── Training Loop ──
    best_val_loss = float("inf")
    best_epoch = 0
    best_path = ""
    start = time.time()

    print(f"\n🏋️ Bắt đầu training...\n")
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg)
        val_mae = validate(model, val_loader, criterion, device, cfg)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_mae < best_val_loss:
            best_val_loss = val_mae
            best_epoch = epoch + 1
            best_path = os.path.join(model_dir, "model_best.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch + 1,
                "val_mae": val_mae,
            }, best_path)
            marker = " ← best ✓"

        scheduler.step(val_mae)
        print(f"  Epoch {epoch+1:>3}/{cfg['epochs']} | "
              f"Train MAE: {train_mae:.1f} | Val MAE: {val_mae:.1f} | "
              f"LR: {lr_now:.1e} | {dt:.0f}s{marker}")

    total_time = (time.time() - start) / 60
    print(f"\n{'─' * 60}")
    print(f"  ✅ Training hoàn thành!")
    print(f"  Best Val MAE : {best_val_loss:.1f} (epoch {best_epoch})")
    print(f"  Thời gian    : {total_time:.1f} phút")

    # ── Test ──
    if best_path:
        ckpt = torch.load(best_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    test_mae, mae_by_tc = test_with_breakdown(model, test_loader, criterion, device, cfg)

    print(f"\n📊 Test MAE (overall): {test_mae:.1f}")
    print(f"   MAE theo thể thức:")
    for tc, mae in sorted(mae_by_tc.items(), key=lambda x: -x[1]):
        print(f"     {tc:>14}: {mae:.1f}")
    print(f"{'═' * 60}")

    return model, test_mae, mae_by_tc

# %%
# 🚀 CHẠY TRAINING
model, test_mae, mae_by_tc = main(CONFIG)

# %% [markdown]
# ## 7. Ablation: So sánh có/không CPL
#
# Bỏ comment cell dưới để chạy ablation.

# %%
# # --- ABLATION: Không CPL, không Blunder (= paper gốc) ---
# ablation_config = CONFIG.copy()
# ablation_config["use_cpl"] = False
# ablation_config["use_blunder"] = False
# ablation_config["epochs"] = 60
# _, ablation_mae, _ = main(ablation_config)
# print(f"\n📈 So sánh: Paper gốc MAE = {ablation_mae:.1f} | Cải tiến MAE = {test_mae:.1f}")
