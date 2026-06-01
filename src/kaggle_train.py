import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import polars as pl
import numpy as np
import time
import chess
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ==========================================
# 1. DATA ENCODER & DATASET
# ==========================================
PIECE_TYPE_TO_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None: continue
        plane_idx = PIECE_TYPE_TO_PLANE[piece.piece_type]
        if piece.color == chess.BLACK: plane_idx += 6
        row, col = square // 8, square % 8
        planes[plane_idx, row, col] = 1.0
    return planes

def replay_game_to_boards(moves_san: str, max_moves=150) -> list:
    board = chess.Board()
    boards = []
    tokens = [t for t in moves_san.split() if not (t.endswith('.') or t in ('1-0','0-1','1/2-1/2','*'))]
    for token in tokens[:max_moves]:
        try:
            board.push_san(token)
            boards.append(encode_board(board))
        except: break
    return boards

class KaggleChessDataset(Dataset):
    def __init__(self, df_subset, max_moves=100):
        self.df = df_subset
        self.max_moves = max_moves
        self.ratings_mean, self.ratings_std = 1514.0, 366.0
        self.clocks_mean, self.clocks_std = 273.0, 380.0
        self.cpl_mean, self.cpl_std = 50.0, 100.0

    def __len__(self):
        return self.df.height

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        
        # 1. Positions (12x8x8)
        board_arrays = replay_game_to_boards(row["Moves"], self.max_moves)
        if not board_arrays: board_arrays = [np.zeros((12,8,8), dtype=np.float32)]
        length = len(board_arrays)
        positions = torch.tensor(np.stack(board_arrays), dtype=torch.float32)

        # 2. Clocks (Chuẩn hóa) - MOCK if not exists
        clocks = torch.zeros(length, dtype=torch.float32) # Thay bằng parse JSON nếu có ClockSeq

        # 3. CPL & Blunders - MOCK if not exists
        cpls = torch.zeros(length, dtype=torch.float32)
        blunders = torch.zeros(length, dtype=torch.float32)

        # 4. Targets (ELO)
        white_elo, black_elo = float(row["WhiteElo"]), float(row["BlackElo"])
        targets = torch.tensor([white_elo, black_elo], dtype=torch.float32)
        targets = (targets - self.ratings_mean) / self.ratings_std

        return {'positions': positions, 'clocks': clocks, 'cpls': cpls, 
                'blunders': blunders, 'targets': targets, 'length': length}

def collate_fn(batch):
    positions = pad_sequence([item['positions'] for item in batch], batch_first=True)
    clocks = pad_sequence([item['clocks'] for item in batch], batch_first=True)
    cpls = pad_sequence([item['cpls'] for item in batch], batch_first=True)
    blunders = pad_sequence([item['blunders'] for item in batch], batch_first=True)
    targets = torch.stack([item['targets'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.int)
    return {'positions': positions, 'clocks': clocks, 'cpls': cpls, 
            'blunders': blunders, 'targets': targets, 'lengths': lengths}

# ==========================================
# 2. KIẾN TRÚC RATING NET (CNN + BiLSTM)
# ==========================================
class KaggleRatingNet(nn.Module):
    def __init__(self, conv_filters=32, lstm_layers=2, lstm_h=64, fc_h=32, dropout=0.5):
        super().__init__()
        # CNN trích xuất bàn cờ
        self.conv1 = nn.Conv2d(12, conv_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters*2, 3, padding=1)
        self.conv3 = nn.Conv2d(conv_filters*2, conv_filters*4, 3, padding=1)
        self.pool = nn.AvgPool2d(2, 2)
        
        # BiLSTM kết hợp CNN Vector + Clock + CPL + Blunder
        cnn_out_dim = conv_filters * 4 * 1 * 1 # (Sau 3 lần pool 8x8 -> 1x1)
        lstm_input_dim = cnn_out_dim + 3 # +1 Clock, +1 CPL, +1 Blunder
        
        self.lstm = nn.LSTM(lstm_input_dim, lstm_h, num_layers=lstm_layers, 
                            batch_first=True, bidirectional=True)
        
        # Output Layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_h * 2, fc_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_h, 2) # Dự đoán 2 số: ELO Trắng & ELO Đen
        )

    def forward(self, positions, clocks, cpls, blunders, lengths):
        B, T = positions.shape[:2]
        x = positions.view(-1, 12, 8, 8)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(B, T, -1) # Vector hóa ảnh CNN
        
        # Nối (Concatenate) TẤT CẢ các features vào chuỗi
        lstm_in = torch.cat((x, clocks.unsqueeze(2), cpls.unsqueeze(2), blunders.unsqueeze(2)), dim=2)
        
        packed_in = pack_padded_sequence(lstm_in, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_in)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # Trích xuất state ở bước thời gian (nước đi) cuối cùng
        last_out = lstm_out[torch.arange(B), lengths - 1, :]
        return self.fc(last_out)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_model(data_path="sample_30k_dl.parquet"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Đọc Parquet (Kaggle chỉ cần up file này lên)
    df = pl.read_parquet(data_path)
    train_df, val_df = df[:int(0.8*df.height)], df[int(0.8*df.height):]
    
    train_loader = DataLoader(KaggleChessDataset(train_df), batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(KaggleChessDataset(val_df), batch_size=32, collate_fn=collate_fn)
    
    model = KaggleRatingNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss() # Dùng MAE (L1 Loss)
    
    for epoch in range(10):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch['positions'].to(device), batch['clocks'].to(device), 
                        batch['cpls'].to(device), batch['blunders'].to(device), batch['lengths'])
            loss = criterion(out, batch['targets'].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    # Trên Kaggle, bạn đổi đường dẫn này thành "/kaggle/input/chess-data/sample_30k.parquet"
    # train_model("đường_dẫn_tới_file.parquet")
    pass
