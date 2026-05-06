"""RatingNet — CNN + Bi-LSTM cho dự đoán ELO.

Mở rộng từ paper arXiv:2409.11506:
  - CNN 4 tầng: Board State (12×8×8) → feature vector
  - Concat: [cnn_out, clock, cpl, blunder] tại mỗi time-step
  - Bi-LSTM: Sequence → per-step hidden states
  - FC: Last time-step → [WhiteELO, BlackELO]

Input LSTM = conv_filters*8 + 3 (clock + cpl + blunder)
  (Paper gốc chỉ có +1 cho clock, chúng ta thêm +2 cho cpl và blunder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RatingNet(nn.Module):
    """Mô hình dự đoán ELO — CNN + Bi-LSTM.

    Kiến trúc CNN giữ nguyên từ paper (4 tầng Conv2D + BN + AvgPool).
    Bổ sung thêm CPL + Blunder vào LSTM input so với paper gốc.
    """

    def __init__(
        self,
        conv_filters: int = 32,
        lstm_layers: int = 3,
        lstm_hidden: int = 64,
        fc1_hidden: int = 32,
        dropout_rate: float = 0.5,
        bidirectional: bool = True,
        use_cpl: bool = True,
        use_blunder: bool = True,
    ):
        """
        Args:
            conv_filters: Số filter tầng Conv2D đầu tiên (tầng sau nhân đôi).
            lstm_layers: Số lớp LSTM.
            lstm_hidden: Kích thước hidden state LSTM.
            fc1_hidden: Kích thước FC layer đầu tiên.
            dropout_rate: Tỉ lệ dropout.
            bidirectional: Dùng Bi-LSTM hay Uni-LSTM.
            use_cpl: Có dùng CPL feature không.
            use_blunder: Có dùng Blunder flag không.
        """
        super().__init__()

        self.use_cpl = use_cpl
        self.use_blunder = use_blunder

        # ── CNN Block: 4 tầng Conv2D ──
        # Input: [B*T, 12, 8, 8] → Output: [B*T, conv_filters*8, 1, 1]
        self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters)

        self.conv2 = nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters * 2)

        self.conv3 = nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_filters * 4)

        self.conv4 = nn.Conv2d(conv_filters * 4, conv_filters * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_filters * 8)

        self.pool = nn.AvgPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # ── LSTM ──
        # CNN output: conv_filters * 8 (sau 3 lần pool 8→4→2→1, flatten = conv_filters*8)
        # + 1 (clock) + 1 (cpl, optional) + 1 (blunder, optional)
        extra_features = 1  # clock (luôn có)
        if use_cpl:
            extra_features += 1
        if use_blunder:
            extra_features += 1

        lstm_input_size = conv_filters * 8 + extra_features

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # ── FC Head ──
        fc_input_size = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.fc1 = nn.Linear(fc_input_size, fc1_hidden)
        self.fc2 = nn.Linear(fc1_hidden, 2)  # Output: [WhiteELO, BlackELO]

    def forward(
        self,
        positions: torch.Tensor,
        clocks: torch.Tensor,
        lengths: torch.Tensor,
        cpls: torch.Tensor | None = None,
        blunders: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            positions: [B, T, 12, 8, 8] — Board states
            clocks:    [B, T]            — Clock times (chuẩn hóa)
            lengths:   [B]              — Độ dài thực tế mỗi sequence
            cpls:      [B, T]            — CPL per move (chuẩn hóa, optional)
            blunders:  [B, T]            — Blunder flags (optional)

        Returns:
            all_outputs: [B, T, 2] — Predictions tại mọi time-step
            last_output: [B, 2]    — Prediction tại time-step cuối
        """
        batch_size = positions.size(0)
        seq_len = positions.size(1)

        # ── CNN: xử lý tất cả positions cùng lúc ──
        # Reshape: [B, T, 12, 8, 8] → [B*T, 12, 8, 8]
        x = positions.view(-1, 12, 8, 8)

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)   # 8→4

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)   # 4→2

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)   # 2→1

        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        # Flatten spatial dims: [B*T, C, 1, 1] → [B, T, C]
        x = x.view(batch_size, seq_len, -1)

        # ── Concat extra features ──
        features_to_cat = [x, clocks.unsqueeze(2)]  # clock: [B, T, 1]

        if self.use_cpl and cpls is not None:
            features_to_cat.append(cpls.unsqueeze(2))  # [B, T, 1]

        if self.use_blunder and blunders is not None:
            features_to_cat.append(blunders.unsqueeze(2))  # [B, T, 1]

        lstm_input = torch.cat(features_to_cat, dim=2)
        # [B, T, conv_filters*8 + extra_features]

        # ── LSTM ──
        # Pack để bỏ qua padding
        packed_input = pack_padded_sequence(
            lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # ── FC Head ──
        y = F.leaky_relu(self.fc1(lstm_output))
        y = self.dropout(y)
        all_outputs = self.fc2(y)  # [B, T, 2]

        # Lấy output tại time-step cuối cùng của mỗi sequence
        idx = torch.arange(batch_size, device=positions.device)
        last_output = all_outputs[idx, lengths - 1, :]  # [B, 2]

        return all_outputs, last_output
