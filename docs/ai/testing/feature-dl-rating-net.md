---
phase: testing
title: "DL Rating Net — Testing Strategy"
description: >
  Chiến lược kiểm thử cho pipeline DL Rating Net.
date: 2026-04-27
---

# DL Rating Net — Testing Strategy

## Test Coverage Goals

- Unit test: BoardEncoder, ChessDataset, RatingNet forward pass
- Integration test: Full pipeline từ parquet → training step
- Smoke test: Training convergence trên tập nhỏ (100 ván)

## Unit Tests

### BoardEncoder
- [ ] Test encode vị trí bắt đầu (starting position) → shape `[12, 8, 8]`
- [ ] Test encode sau `1. e4` → pawn trắng di chuyển đúng plane
- [ ] Test tổng số quân = 32 ở starting position
- [ ] Test encode FEN tùy ý → khớp với expected planes

### ChessDataset
- [ ] Test `__len__` khớp số ván trong parquet
- [ ] Test `__getitem__` trả về dict đúng keys
- [ ] Test padding: ván ngắn → pad zeros, ván dài → truncate
- [ ] Test collate_fn: batch tensor shapes đúng

### RatingNet
- [ ] Test forward pass: input shape → output shape `[B, 2]`
- [ ] Test forward pass với bidirectional=True/False
- [ ] Test gradient flow: loss.backward() không lỗi

## Integration Tests

- [ ] Full pipeline: load 10 ván → Dataset → DataLoader → 1 training step
- [ ] Verify loss giảm sau vài epoch trên tập nhỏ

## Smoke Test Commands

```bash
conda activate MMDS

# Unit tests
pytest tests/test_board_encoder.py -v
pytest tests/test_chess_dataset.py -v
pytest tests/test_rating_net.py -v

# Smoke training (100 ván, 5 epochs)
python -m src.models.train --smoke-test
```

## Performance Testing

- [ ] Measure DataLoader throughput (ván/giây)
- [ ] Measure training step time (giây/batch)
- [ ] Monitor GPU VRAM usage
