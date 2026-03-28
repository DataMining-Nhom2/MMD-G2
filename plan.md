# 📜 ĐẠI KẾ HOẠCH NÂNG CẤP LÒ BÁT QUÁI STOCKFISH (FEATURE ENGINEERING V2)

**Mục tiêu:** Vắt kiệt sức mạnh của Stockfish, chuyển đổi từ việc chỉ đo CPL thô sơ sang đo lường toàn diện "Trình độ thực chiến" (Performance Rating) của kỳ thủ thông qua phân tích khía cạnh chuyên sâu (Giai đoạn ván cờ, Xác suất WDL, Độ khớp máy). Tuyệt đối không dùng dữ liệu rò rỉ (Leakage).

---

## PHẦN 1: THANH LỌC DỮ LIỆU THÔ (RAW DATA)
*Quy tắc thép: Loại bỏ mọi tính năng mang tính "Gợi ý thủ thuật" để ép Model phải "nhìn" vào cờ vua đích thực.*

### 1.1 Những Cột Bị Trảm Bỏ Khốc Liệt 🔪
- `WhiteElo`, `BlackElo`, `EloAvg`: Biến mục tiêu (Target). Dùng là ăn gian.
- `WhiteRatingDiff`, `BlackRatingDiff`: Kết quả sau trận (Leakage).
- `Result`, `ResultNumeric`: Thắng bại ván đơn không định nghĩa được Kiện tướng.
- `BaseTime`, `Increment`, `GameFormat`: Tội lỗi lớn nhất của mô hình. Nếu đưa vào, máy sẽ chỉ đoán mò ELO dựa trên thể thức cờ chớp/cờ chậm thay vì phân tích nước đi.
- `Termination`: Giống hệt thời gian, nếu thua vì hết giờ thường là cờ chớp. Đã loại trừ.

### 1.2 Nhóm Vàng Ròng Giữ Lại 🥇
Chỉ giữ lại 2 trụ cột từ Raw Data phản ánh **Phong cách và Thời lượng**:
1. **`ECO` & `EcoCategory`**: Mã khai cuộc (One-hot top 100). Dấu hiệu mạnh nhất để rà quét học vẹt vs hiểu biết lý thuyết (A00 vs C60).
2. **`NumMoves_Norm`**: Số lượng nước đi (Chuẩn hóa). Ván cờ nhây nhớt 80 nước tàn cuộc khác bọt hoàn toàn với bị lừa đúp cờ trong 10 nước.

---

## PHẦN 2: BẢNG PHONG THẦN STOCKFISH FEATURES (V2)
Yêu cầu mã nguồn `StockfishTransformer` phải khạc ra được binh đoàn 11 Features sau đây cho mỗi ván cờ.

### Nhóm A: Chỉ số Toàn ván (Tỷ lệ hóa)
Thay vì đếm số đếm thuần túy (dễ bị thiên vị bởi ván dài), chúng ta sẽ chuyển sang Tỷ lệ %.
1. `avg_cpl`: Trung bình hụt điểm Centipawn toàn ván (Giữ nguyên).
2. `cpl_std`: Độ lệch chuẩn của CPL chặn bắt những gã múa cờ (Lên đồng thất thường).
3. `blunder_rate`: (Số nước Blunder > 300) / Tổng số nước.
4. `mistake_rate`: (Số nước Mistake 100-300) / Tổng số nước.
5. `inaccuracy_rate`: (Số nước Inaccuracy 50-100) / Tổng số nước.

### Nhóm B: Chỉ số Phân Nhánh Giai Đoạn (ĐỘT PHÁ 1)
Bóc trần sự thật "Học vẹt khai cuộc nhưng mù tàn cuộc". Phân tích CPL theo 3 Phase.
6. `opening_cpl`: Trung bình CPL 10 nước đầu (Ply 1 -> 20).
7. `midgame_cpl`: Trung bình CPL nước 11 đến 30 (Ply 21 -> 60). *(Nếu ván cờ kết thúc sớm hơn 11 nước -> gán `NaN`)*.
8. `endgame_cpl`: Trung bình CPL từ nước 31 trở đi (Ply 61+). *(Nếu ván cờ kết thúc sớm hơn 31 nước -> gán `NaN`)*.
> **Lưu ý Quân Sự:** Để nguyên giá trị `NaN` (Missing Value) cho các ván quá ngắn. XGBoost / LightGBM cực kỳ khát máu với `NaN` và tự coi đó là một Dấu hiệu (Feature) phản ánh ván cờ bị huỷ diệt sớm!

### Nhóm C: Chỉ số Cấp Độ Đấng (WDL & PV Match) (ĐỘT PHÁ 2)
CPL bị yếu điểm khi lợi thế quá lớn (Biếu nốt con Xe khi đã hơn Hậu). Cần xác suất Thắng (WDL) để đo lường.
9. `avg_wdl_loss`: Trung bình độ đánh rơi XÁC SUẤT THẮNG (% Win Rate drop). Một nước cờ ngu ném đi 30% cơ hội thắng quan trọng hơn mất 3 phân Tốt.
10. `max_wdl_loss`: Pha bóp dái chí mạng nhất ván cờ (Nước cờ ném đi nhiều % Thắng nhất).
11. `best_move_match_rate`: Tỷ lệ phần trăm (%) nước đi đời thực **trùng khớp 100%** với Nước Số 1 (Top 1 PV) mà Stockfish đề xuất. (Vũ khí phát hiện Đại kiện tướng).

---

## PHẦN 3: KẾ HOẠCH HÀNH ĐỘNG CỤ THỂ (CODE IMPLEMENTATION)

- [ ] **Bước 1:** Bật cờ `UCI_LimitStrength` = `False` và đảm bảo Stockfish 16+ đang bật mạng Nơ-ron (NNUE) để có thể phun ra chỉ số WDL (Win/Draw/Loss probability).
- [ ] **Bước 2:** Cập nhật file `src/feature_config.py`. Thêm các hằng số định nghĩa ngưỡng Giai đoạn (Ngưỡng 10 nước, 30 nước).
- [ ] **Bước 3:** Đập đi xây lại hàm `analyze_game()` trong class `StockfishTransformer` (file `src/feature_engineering.py`):
    - Khi lặp từng `move` trong ván cờ, lấy thông tin đa chiều: `.score(mate_score=...)`, và WDL `.wdl(pov=...)`.
    - Kiểm tra xem nước cờ người chơi vừa đi có nằm ở index [0] trong chuỗi PV của Stockfish không để cộng điểm `Best_Move`.
    - Phân mảnh list CPL/WDL thu thập được thành 3 list theo Index (Opening, Mid, End) rồi tính trung bình. Nếu rỗng mảng -> điền rỗng (NaN).
- [ ] **Bước 4:** Chạy lại luồng Pipeline trên tập nhỏ 30k ván. Xuất File Parquet phiên bản V2 mới.
- [ ] **Bước 5:** Bơm sang vòng huấn luyện `notebooks/stockfish-baseline.ipynb`, kiểm nghiệm mô hình XGBoost. Bật đồ thị `Feature_Importance` xem đám `wdl_loss` và `best_move_rate` có đá bay CPL thô xuống đáy bảng xếp hạng hay không! Lên được >50% Accuracy coi như đại công cáo thành!
