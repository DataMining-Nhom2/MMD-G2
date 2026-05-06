"""
Unit tests cho Clock Service
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_server.clock import ClockService, ClockState


class TestClockInit:
    """Test khởi tạo clock"""

    def test_parse_time_control_15_0(self):
        """Parse 15+0"""
        service = ClockService()
        seconds, increment = service.parse_time_control("15+0")
        assert seconds == 900.0
        assert increment == 0.0

    def test_parse_time_control_5_3(self):
        """Parse 5+3"""
        service = ClockService()
        seconds, increment = service.parse_time_control("5+3")
        assert seconds == 300.0
        assert increment == 3.0

    def test_parse_time_control_invalid(self):
        """Parse invalid format - fallback to default"""
        service = ClockService()
        seconds, increment = service.parse_time_control("invalid")
        assert seconds == 900.0
        assert increment == 0.0


class TestClockCreation:
    """Test tạo clock"""

    def test_create_clock_default(self):
        """Tạo clock với default"""
        service = ClockService()
        clock = service.create_clock("ROOM001")

        assert clock.white_remaining == 900.0
        assert clock.black_remaining == 900.0
        assert clock.increment == 0.0
        assert clock.current_turn == "white"

    def test_create_clock_custom(self):
        """Tạo clock với custom time control"""
        service = ClockService()
        clock = service.create_clock("ROOM002", "5+3")

        assert clock.white_remaining == 300.0
        assert clock.black_remaining == 300.0
        assert clock.increment == 3.0


class TestClockRecording:
    """Test ghi nhận thời gian"""

    def test_record_time(self):
        """Test ghi nhận thời gian"""
        service = ClockService()
        service.create_clock("ROOM003")

        # Record some times
        service.record_time("ROOM003", "white", 5.0)
        service.record_time("ROOM003", "black", 3.0)

        times = service.get_clock_times("ROOM003")
        assert len(times) == 2
        assert 5.0 in times
        assert 3.0 in times


class TestClockMakeMove:
    """Test make_move"""

    def test_make_move_white(self):
        """Test make_move cho white"""
        service = ClockService()
        service.create_clock("ROOM004", "15+0")
        service.start_clock("ROOM004")

        time.sleep(0.1)  # Small delay
        elapsed = service.make_move("ROOM004", "white")

        assert elapsed is not None
        assert elapsed > 0
        assert elapsed < 1  # Should be less than 1 second

        times = service.get_clock_times("ROOM004")
        assert len(times) == 1

        # Check current turn switched
        clock = service.get_clock("ROOM004")
        assert clock.current_turn == "black"

    def test_make_move_black(self):
        """Test make_move cho black"""
        service = ClockService()
        service.create_clock("ROOM005", "15+0")
        service.start_clock("ROOM005")

        # White moves first
        service.make_move("ROOM005", "white")
        time.sleep(0.1)

        # Black moves
        elapsed = service.make_move("ROOM005", "black")
        assert elapsed is not None

        times = service.get_clock_times("ROOM005")
        assert len(times) == 2


class TestClockTimeout:
    """Test timeout detection"""

    def test_no_timeout_initially(self):
        """Không có timeout lúc đầu"""
        service = ClockService()
        service.create_clock("ROOM006", "15+0")

        timeout = service.is_timeout("ROOM006")
        assert timeout is None

    def test_timeout_white(self):
        """Timeout cho white"""
        service = ClockService()
        service.create_clock("ROOM007", "1+0")  # 1 second
        service.start_clock("ROOM007")

        # Manually set white remaining to 0
        clock = service.get_clock("ROOM007")
        clock.white_remaining = 0

        timeout = service.is_timeout("ROOM007")
        assert timeout == "white"


class TestClockDelete:
    """Test xóa clock"""

    def test_delete_clock(self):
        """Test xóa clock"""
        service = ClockService()
        service.create_clock("ROOM008")

        assert service.get_clock("ROOM008") is not None

        service.delete_clock("ROOM008")
        assert service.get_clock("ROOM008") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
