---
phase: implementation
title: V3 Regression Evaluation Report - ELO Continuous Prediction
description: Phân tích chuyên sâu kết quả Regression V3, giải thích hiện tượng Đa cộng tuyến và Sai số lân cận (Adjacent Misclassification).
---

# Báo cáo Đánh giá Mô hình XGBoost Regression V3

## 1. Mục tiêu và Kết quả Tổng quan
Chuyển đổi bài toán từ Phân lớp nhãn cứng (Classification 5 Bands) sang Dự đoán ELO liên tục (Regression) sử dụng 11 features trích xuất từ Stockfish V2.

**Kết quả Full Model (5-Fold CV):**
- **MAE (Mean Absolute Error):** 247.8 ELO (±2.1)
- **RMSE:** 315.8 ELO (±2.3)
- **R² Score:** 0.6535
- **Đánh giá:** 🟡 Khá khả quan (Sát mốc mục tiêu ≤ 220). Mô hình đã giải thích được ~65.3% sự biến thiên của ELO con người dựa trên các con số trừu tượng do AI (Stockfish) nội suy.

---

## 2. Vai trò của Stockfish Features (Ablation Study)
Dù biểu đồ Feature Importance Top 20 bị áp đảo bởi các mã khai cuộc ECO (vì đây là các One-Hot Variables chẻ nhánh cục bộ dễ sinh Gain), sức mạnh thực sự nằm ở 3 cụm Stockfish:

1. **Tabular Only (Không Engine):** MAE = 372.4 (Mô hình gần như mù mờ).
2. **Thêm Nhóm A (CPL tổng thể):** MAE giảm mạnh 100 điểm xuống 273.8.
3. **Thêm Nhóm B (CPL Khai/Trung/Tàn):** Giảm tiếp MAE xuống 260.1.
4. **Thêm Nhóm C (WDL Loss & Best Move Match):** Chốt mốc MAE 247.8 (Giảm thêm 12.3 ELO).

**Nhận định Đa cộng tuyến (Collinearity):**  
WDL Loss và Best Move Match đóng góp thực tế là **rất lớn** (cứu được 12 ELO), nhưng bị đẩy xuống dưới bảng Top 20 vì lượng thông tin phân loại cốt lõi đã bị `avg_cpl` (Centipawn Loss) "hút" sạch ở các node gốc của Decision Tree. WDL chỉ giữ vai trò Fine-tune ở các nhánh sâu.

---

## 3. Lập luận Bảo vệ Tỷ lệ Classification (43.57%)
Khi ánh xạ ngược ELO dự đoán (trung bình 1588) về 5 Bands của V2, Accuracy tụt từ 47.59% xuống 43.57%. Tuy nhiên, đây **không phải là bước lùi**, mà là minh chứng cho sự "thông minh thực tế" của Regression:

- **Ranh giới cực nhạy:** Các Band chỉ rộng 400 ELO. Một sai số MAE ~248 ELO là quá đủ để đẩy một kỳ thủ `1750` (Advanced) sang `1850` (Expert), đánh dấu là "Sai" trong Classification.
- **Sai số Lân cận (Adjacent Misclassification):** Nhìn vào Confusion Matrix, mô hình hầu như **chỉ sai số đúng 1 bậc**. 
    - Lớp `Master` (>2200) bị phân nhầm thành `Expert` (1800-2200) tới 3050 lần, nhưng không bao giờ bị xếp vào `Beginner`.
    - Lớp `Beginner` (<1000) bị phân nhầm thành `Intermediate` (1000-1400) tới 2953 lần, nhưng tuyệt đối không bị xếp vào `Expert` hay `Master`.
- **Ordinal Error:** Về mặt xếp hạng thứ tự, V3 Regression an toàn và logic hơn hẳn V2 Classification (vì nó hiểu tính chất "gần đúng", trong khi V2 coi điểm biên giới là bức tường cứng).

---

## 4. Hướng tối ưu tiếp theo (Tuning & V4)

Để bóp MAE từ 247.8 xuống mốc tối thượng ≤ 220, chúng ta có 2 hướng đi chính:

### 4.1. Fine-tuning Hyperparameters (XGBoost)
Mô hình hiện tại đang chạy các tham số tĩnh (`max_depth=6`, `learning_rate=0.1`, `n_estimators=300`). Để Fine-tune và ép MAE xuống, ta có thể dùng **Optuna** hoặc **GridSearchCV** dò tìm bộ thông số sau:
- Tăng `n_estimators` lên `1000` - `2000` và giảm `learning_rate` xuống `0.01` - `0.05` để XGBoost trau chuốt các cụm dữ liệu viền.
- Tinh chỉnh `colsample_bytree` và `subsample` để giảm Overfitting do 100+ đặc trưng phân loại ECO gây ra.
- Tùy chỉnh `gamma` và `min_child_weight` để phạt nặng các nhánh rẽ quá sâu lẻ tẻ.

*(Sự tối ưu này có thể kéo thêm khoảng 10-20 ELO cho MAE).*

### 4.2. Deep Learning với Chuỗi Thời gian (Sequence CPL) -> Phase V4
XGBoost đại diện cho các phương pháp học máy dạng bảng (Tabular). Mặc dù rất tốt, nhưng việc chúng ta cộng dồn diễn biến của cả ván cờ (70-80 nước) thành vài con số tĩnh (Trung bình, Độ lệch chuẩn, Tổng số lỗi) đã **vứt bỏ hoàn toàn tính tuần tự của thời gian**.
Ví dụ: Tâm lý của người chơi sẽ sụp đổ sau khi dính 1 Blunder chí mạng ở nước thứ 40, và các nước sau đó sẽ tràn ngập Inaccuracy.
Việc đưa trực tiếp danh sách mảng số `[cpl_nước1, cpl_nước2, cpl_nước3...]` thành Time-Series vào các thiết kế Học sâu tiên tiến như **Transformer**, **1D-CNN**, hay **LSTM** mới là "quân cờ bí mật" để giải mã triệt để mức ELO của con người.
