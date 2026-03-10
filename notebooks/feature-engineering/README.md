# Notebooks — Feature Engineering

Mục đích:

- Lưu notebook thử nghiệm trực quan cho Feature Engineering.
- Theo dõi quick checks mà không mất context như chạy lệnh dài trên terminal.

## Cấu trúc đề xuất

- `exploration/`: notebook EDA/quan sát phân phối feature.
- `pipeline-smoke/`: notebook kiểm tra nhanh pipeline theo sample nhỏ.
- `rebaseline/`: notebook train/eval/ablation và biểu đồ importance.

## Gợi ý workflow

1. Dùng sample nhỏ (ví dụ 3k-10k rows) để chạy nhanh.
2. Lưu output tạm trong `data/features/smoke/`.
3. Chỉ khi pass smoke mới chạy full pipeline.

## Rule bắt buộc

- **Notebook-first cho test/thử nghiệm**: mọi kiểm tra exploratory, smoke run, ablation, và phân tích metric phải thực hiện trong notebook trước.
- Terminal chỉ dùng cho:
  - chạy test tự động (pytest),
  - tác vụ batch/full run đã chốt,
  - tác vụ CI/CD hoặc automation không phù hợp notebook.

## Lưu ý

- Chọn đúng kernel conda `MMDS`.
- Ghi lại seed, tham số và metric trong từng notebook.
