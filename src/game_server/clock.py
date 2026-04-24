"""
Clock Service - Quản lý đồng hồ thi đấu
Ghi nhận thời gian suy nghĩ cho từng nước đi
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClockState:
    """Trạng thái đồng hồ"""
    white_remaining: float  # Giây còn lại
    black_remaining: float
    clock_times: list[float] = field(default_factory=list)  # Thời gian suy nghĩ mỗi nước
    current_turn: str = "white"  # Lượt đi hiện tại
    last_move_time: float = field(default_factory=time.time)  # Thời điểm bắt đầu lượt
    increment: float = 0.0  # Increment sau mỗi nước (giây)
    is_running: bool = False


class ClockService:
    """Service quản lý đồng hồ"""

    def __init__(self):
        self.clocks: dict[str, ClockState] = {}

    def parse_time_control(self, time_control: str) -> tuple[float, float]:
        """
        Parse time control string
        VD: "15+0" -> (900.0, 0.0), "5+3" -> (300.0, 3.0)
        """
        try:
            parts = time_control.split('+')
            minutes = float(parts[0])
            increment = float(parts[1]) if len(parts) > 1 else 0.0
            return minutes * 60.0, increment
        except (ValueError, IndexError):
            return 900.0, 0.0  # Default 15 minutes

    def create_clock(self, room_code: str, time_control: str = "15+0") -> ClockState:
        """
        Tạo đồng hồ mới cho phòng
        """
        total_seconds, increment = self.parse_time_control(time_control)

        clock = ClockState(
            white_remaining=total_seconds,
            black_remaining=total_seconds,
            increment=increment,
            last_move_time=time.time()
        )

        self.clocks[room_code] = clock
        return clock

    def get_clock(self, room_code: str) -> Optional[ClockState]:
        """Lấy trạng thái đồng hồ"""
        return self.clocks.get(room_code)

    def start_clock(self, room_code: str):
        """Bắt đầu đếm đồng hồ"""
        if room_code in self.clocks:
            self.clocks[room_code].is_running = True
            self.clocks[room_code].last_move_time = time.time()

    def stop_clock(self, room_code: str) -> Optional[float]:
        """
        Dừng đồng hồ và ghi nhận thời gian
        Returns: Thời gian đã dùng (giây)
        """
        if room_code not in self.clocks:
            return None

        clock = self.clocks[room_code]
        if not clock.is_running:
            return None

        elapsed = time.time() - clock.last_move_time

        # Trừ thời gian từ người đang đi
        if clock.current_turn == "white":
            clock.white_remaining -= elapsed
        else:
            clock.black_remaining -= elapsed

        # Không âm
        if clock.white_remaining < 0:
            clock.white_remaining = 0
        if clock.black_remaining < 0:
            clock.black_remaining = 0

        clock.is_running = False

        return elapsed

    def record_time(self, room_code: str, color: str, elapsed: float):
        """
        Ghi nhận thời gian suy nghĩ
        Args:
            elapsed: Thời gian đã dùng (giây)
        """
        if room_code in self.clocks:
            self.clocks[room_code].clock_times.append(round(elapsed, 1))

    def switch_turn(self, room_code: str):
        """
        Chuyển lượt và bắt đầu đồng hồ cho người tiếp theo
        """
        if room_code not in self.clocks:
            return

        clock = self.clocks[room_code]

        # Chuyển lượt
        clock.current_turn = "black" if clock.current_turn == "white" else "white"

        # Thêm increment cho lượt mới
        if clock.increment > 0:
            if clock.current_turn == "white":
                clock.white_remaining += clock.increment
            else:
                clock.black_remaining += clock.increment

        # Bắt đầu đếm
        clock.last_move_time = time.time()
        clock.is_running = True

    def make_move(self, room_code: str, color: str) -> Optional[float]:
        """
        Xử lý khi người chơi đi nước
        Returns: Thời gian suy nghĩ
        """
        if room_code not in self.clocks:
            return None

        clock = self.clocks[room_code]

        # Dừng đồng hồ
        elapsed = self.stop_clock(room_code)
        if elapsed is None:
            elapsed = 0

        # Ghi nhận thời gian
        self.record_time(room_code, color, elapsed)

        # Chuyển lượt
        self.switch_turn(room_code)

        return elapsed

    def get_remaining(self, room_code: str) -> Optional[dict]:
        """Lấy thời gian còn lại"""
        if room_code not in self.clocks:
            return None

        clock = self.clocks[room_code]
        return {
            "white": clock.white_remaining,
            "black": clock.black_remaining
        }

    def get_clock_times(self, room_code: str) -> list[float]:
        """Lấy danh sách thời gian suy nghĩ"""
        if room_code in self.clocks:
            return self.clocks[room_code].clock_times
        return []

    def is_timeout(self, room_code: str) -> Optional[str]:
        """
        Kiểm tra có ai hết giờ không
        Returns: "white", "black", hoặc None
        """
        if room_code not in self.clocks:
            return None

        clock = self.clocks[room_code]

        if clock.white_remaining <= 0:
            return "white"
        if clock.black_remaining <= 0:
            return "black"

        return None

    def delete_clock(self, room_code: str):
        """Xóa đồng hồ"""
        if room_code in self.clocks:
            del self.clocks[room_code]


# Singleton instance
clock_service = ClockService()
