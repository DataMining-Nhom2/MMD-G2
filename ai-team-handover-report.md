# Bàn Giao Web Game Server (PoC v6)

Hi Leader & anh em AI team,

Team Web đã xử lý xong phần việc của bên mình theo như doc kiến trúc `v6-poc-system-architecture.md`. File này tóm tắt lại các đầu việc đã done và hướng dẫn anh em cách kéo code về chạy thử, cũng như chuẩn bị sẵn sàng để nối API nhé.

---

## 1. Cách chạy Repo (Local)
Code web nằm trong thư mục `Whess/`. Anh em làm theo các bước sau để chạy cả frontend và backend cùng lúc:

1. Mở terminal, vào thư mục gốc `Whess/`:
   ```bash
   cd Whess
   ```
2. Cài đặt các package cần thiết cho cả thư mục gốc, server và client:
   ```bash
   npm run install:all
   ```
3. Khởi động hệ thống:
   ```bash
   npm run dev
   ```
   *Lệnh này sẽ tự động bật backend ở `http://localhost:3001` và frontend ở `http://localhost:3000`.*

Vào trình duyệt mở `http://localhost:3000` là chơi được ngay, không cần cài cắm db gì đâu nhé.

---

## 2. Những hạng mục đã hoàn thành (Done Checklist)
Team Web đã test kỹ và pass hết các luồng này:
- [x] Tạo phòng có mã riêng, sinh link mời share cho người khác được.
- [x] Người thứ 2 vào phòng tự động được ghép vào phe Đen.
- [x] Đánh cờ mượt mà, đồng bộ qua WebSocket gần như không có độ trễ (< 1s).
- [x] Đồng hồ chạy chuẩn: ông nào đi xong thì dừng, đồng hồ đối thủ chạy. Hết giờ là tự xử thua.
- [x] Checkmate, Xin thua đều hoạt động đúng. Nút "Chơi Lại" reset game ok.
- [x] Tính chuẩn số giây suy nghĩ (`timeSpent`) cho từng nước đi.
- [x] Xong game hiện Loading "Đang phân tích ELO...", ngầm bắn HTTP POST payload (gồm chuỗi PGN & mảng thời gian) sang server AI.
- [x] Nếu server AI sập hoặc timeout (quá 30s) thì web không crash, vẫn hiện kết quả Thắng/Thua nhưng báo lỗi AI.
- [x] Render ngon lành Result Modal khi có response từ AI (hiện ELO, ECO, CPL, Blunders, Lời nhận xét).
- [x] Dev ghi log đầy đủ ở console (`timeSpent` lúc đi cờ, payload lúc game over) để anh em dễ debug.

---

## 3. Phần việc chuyển giao cho AI Team
Web sẽ luôn call sang endpoint mặc định của AI:
`POST http://localhost:8000/api/predict-elo`

**Data Web sẽ gửi đi lúc game over (anh em check log web server sẽ thấy y xì):**
```json
{
  "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6",
  "clock_times": [5.2, 3.1, 12.0, 8.5, 2.1, 45.3, 3.0, 7.2],
  "result": "1-0",
  "time_control": "5+0"
}
```

**Nhiệm vụ của AI team bây giờ:**
1. Code API endpoint `POST /api/predict-elo` ở cổng `8000` bên Python.
2. Cứ lấy data từ payload kia ném vào model để suy luận ra ELO + gọi LLM giải thích.
3. Nhớ trả về JSON chuẩn theo cấu trúc trong file v6 (ví dụ: `response.data.white_elo`, `eco.code`, `stats`...). Web UI đang chờ đúng format đấy để vẽ biểu đồ với chữ.

Anh em cứ bật web lên, chia ra 2 tab tự đánh nhau một ván cho hết giờ hoặc chiếu bí là sẽ thấy request vã sang cổng 8000 ngay. Có gì lỗi thì ới team Web hỗ trợ nhé!

Thanks!
