"""Training script cho RatingNet (CNN + Bi-LSTM).

Cách chạy:
  conda activate MMDS
  python -m src.models.train

  # Smoke test (100 ván, 5 epochs):
  python -m src.models.train --smoke-test

  # Ablation (không CPL):
  python -m src.models.train --no-cpl

  # Ablation (không Blunder):
  python -m src.models.train --no-blunder
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from src.data.chess_dataset import ChessDataset, collate_chess_batch
from src.models.rating_net import RatingNet


# ══════════════════════════════════════════════════════════
# CẤU HÌNH MẶC ĐỊNH
# ══════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Data
    "data_path": "data/processed/sample_30k_dl.parquet",
    "cpl_path": None,
    "max_moves": 150,

    # Model
    "conv_filters": 32,
    "lstm_layers": 3,
    "lstm_hidden": 64,
    "fc1_hidden": 32,
    "dropout_rate": 0.5,
    "bidirectional": True,
    "use_cpl": True,
    "use_blunder": True,

    # Training
    "batch_size": 32,
    "val_batch_size": 256,
    "epochs": 60,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "patience": 5,
    "lr_factor": 0.5,
    "num_workers": 4,

    # Normalization (từ paper, hoặc tính lại)
    "ratings_mean": 1514.0,
    "ratings_std": 366.0,

    # Output
    "experiment_name": "rating_net_v1",
    "model_dir": "models",
}


# ══════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════

def train_one_epoch(
    model: RatingNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> float:
    """Train 1 epoch, trả về average loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        positions = batch["positions"].to(device)
        clocks = batch["clocks"].to(device)
        lengths = batch["lengths"]
        targets = batch["targets"].to(device)
        cpls = batch["cpls"].to(device) if config["use_cpl"] else None
        blunders = batch["blunders"].to(device) if config["use_blunder"] else None

        optimizer.zero_grad()
        _, last_output = model(positions, clocks, lengths, cpls, blunders)

        # Rescale predictions và targets về ELO gốc trước khi tính loss
        pred_elo = last_output * config["ratings_std"] + config["ratings_mean"]
        true_elo = targets * config["ratings_std"] + config["ratings_mean"]

        loss = criterion(pred_elo, true_elo)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: RatingNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> float:
    """Validate, trả về average loss."""
    model.eval()
    total_loss = 0.0

    for batch in loader:
        positions = batch["positions"].to(device)
        clocks = batch["clocks"].to(device)
        lengths = batch["lengths"]
        targets = batch["targets"].to(device)
        cpls = batch["cpls"].to(device) if config["use_cpl"] else None
        blunders = batch["blunders"].to(device) if config["use_blunder"] else None

        _, last_output = model(positions, clocks, lengths, cpls, blunders)

        pred_elo = last_output * config["ratings_std"] + config["ratings_mean"]
        true_elo = targets * config["ratings_std"] + config["ratings_mean"]

        loss = criterion(pred_elo, true_elo)
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def test_with_breakdown(
    model: RatingNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> tuple[float, dict[str, float]]:
    """Test với MAE breakdown theo thể thức.

    Returns:
        (overall_mae, {time_control: mae})
    """
    model.eval()
    total_loss = 0.0
    loss_by_tc: dict[str, float] = {}
    count_by_tc: dict[str, int] = {}

    for batch in loader:
        positions = batch["positions"].to(device)
        clocks = batch["clocks"].to(device)
        lengths = batch["lengths"]
        targets = batch["targets"].to(device)
        time_controls = batch["time_controls"]
        cpls = batch["cpls"].to(device) if config["use_cpl"] else None
        blunders = batch["blunders"].to(device) if config["use_blunder"] else None

        _, last_output = model(positions, clocks, lengths, cpls, blunders)

        pred_elo = last_output * config["ratings_std"] + config["ratings_mean"]
        true_elo = targets * config["ratings_std"] + config["ratings_mean"]

        loss = criterion(pred_elo, true_elo)
        total_loss += loss.item()

        # MAE per item per time-control
        mae_per_item = torch.abs(pred_elo - true_elo).mean(dim=1)
        for idx, tc in enumerate(time_controls):
            loss_by_tc[tc] = loss_by_tc.get(tc, 0.0) + mae_per_item[idx].item()
            count_by_tc[tc] = count_by_tc.get(tc, 0) + 1

    # Average
    for tc in loss_by_tc:
        if count_by_tc[tc] > 0:
            loss_by_tc[tc] /= count_by_tc[tc]

    return total_loss / len(loader), loss_by_tc


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train RatingNet")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Chạy nhanh trên 100 ván, 5 epochs")
    parser.add_argument("--no-cpl", action="store_true",
                        help="Ablation: không dùng CPL")
    parser.add_argument("--no-blunder", action="store_true",
                        help="Ablation: không dùng Blunder")
    parser.add_argument("--data", type=str, default=None,
                        help="Đường dẫn file parquet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.data:
        config["data_path"] = args.data
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.no_cpl:
        config["use_cpl"] = False
    if args.no_blunder:
        config["use_blunder"] = False
    if args.smoke_test:
        config["epochs"] = 5
        config["experiment_name"] = "smoke_test"

    # ── Setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 64}")
    print(f"  RatingNet Training — {config['experiment_name']}")
    print(f"{'─' * 64}")
    print(f"  Device       : {device}")
    print(f"  Data         : {config['data_path']}")
    print(f"  CPL          : {'✓' if config['use_cpl'] else '✗ (ablation)'}")
    print(f"  Blunder      : {'✓' if config['use_blunder'] else '✗ (ablation)'}")
    print(f"  Epochs       : {config['epochs']}")
    print(f"  Batch size   : {config['batch_size']}")
    print(f"  Bidirectional: {'✓' if config['bidirectional'] else '✗'}")
    print(f"{'═' * 64}\n")

    # ── Load Dataset ──
    dataset = ChessDataset(
        parquet_path=config["data_path"],
        max_moves=config["max_moves"],
        cpl_parquet_path=config["cpl_path"],
    )

    # Split: 70% train, 20% val, 10% test
    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)

    if args.smoke_test:
        train_idx = train_idx[:80]
        val_idx = val_idx[:10]
        test_idx = test_idx[:10]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, collate_fn=collate_chess_batch,
        num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["val_batch_size"],
        shuffle=False, collate_fn=collate_chess_batch,
        num_workers=config["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["val_batch_size"],
        shuffle=False, collate_fn=collate_chess_batch,
        num_workers=config["num_workers"], pin_memory=True,
    )

    # ── Model ──
    model = RatingNet(
        conv_filters=config["conv_filters"],
        lstm_layers=config["lstm_layers"],
        lstm_hidden=config["lstm_hidden"],
        fc1_hidden=config["fc1_hidden"],
        dropout_rate=config["dropout_rate"],
        bidirectional=config["bidirectional"],
        use_cpl=config["use_cpl"],
        use_blunder=config["use_blunder"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params : {total_params:,}")

    # ── Optimizer + Scheduler ──
    criterion = nn.L1Loss()  # MAE
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=config["patience"], factor=config["lr_factor"]
    )

    # ── Model dir ──
    model_dir = Path(config["model_dir"]) / config["experiment_name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Training Loop ──
    best_val_loss = float("inf")
    best_epoch = 0
    best_path = ""
    start_time = time.time()

    print(f"\n  Bắt đầu training...\n")

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss = validate(model, val_loader, criterion, device, config)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch + 1:>3}/{config['epochs']} | "
            f"Train MAE: {train_loss:.1f} | "
            f"Val MAE: {val_loss:.1f} | "
            f"LR: {current_lr:.1e} | "
            f"Time: {epoch_time:.0f}s"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_path = str(model_dir / f"model_best.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, best_path)
            print(f"    → Saved best model (Val MAE: {val_loss:.1f})")

    total_time = time.time() - start_time

    print(f"\n{'─' * 64}")
    print(f"  Training hoàn thành!")
    print(f"  Best Val MAE : {best_val_loss:.1f} (epoch {best_epoch})")
    print(f"  Thời gian    : {total_time / 60:.1f} phút")

    # ── Test ──
    if best_path:
        checkpoint = torch.load(best_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_mae, mae_by_tc = test_with_breakdown(model, test_loader, criterion, device, config)

    print(f"\n  Test MAE (overall): {test_mae:.1f}")
    print(f"  Test MAE theo thể thức:")
    for tc, mae in sorted(mae_by_tc.items(), key=lambda x: -x[1]):
        print(f"    {tc:>14}: {mae:.1f}")
    print(f"{'═' * 64}")


if __name__ == "__main__":
    main()
