"""
Unit tests cho Room Manager
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_server.rooms import RoomManager, Room, RoomStatus, PlayerColor


class TestRoomCodeGeneration:
    """Test tạo mã phòng"""

    def test_generate_room_code_length(self):
        """Mã phòng có 6 ký tự"""
        manager = RoomManager()
        code = manager.generate_room_code()
        assert len(code) == 6

    def test_generate_room_code_alphanumeric(self):
        """Mã phòng là alphanumeric"""
        manager = RoomManager()
        code = manager.generate_room_code()
        assert code.isalnum() is True
        assert code.isupper() is True

    def test_generate_room_code_unique(self):
        """Mã phòng không trùng lặp"""
        manager = RoomManager()
        codes = set()
        for _ in range(100):
            code = manager.generate_room_code()
            assert code not in codes
            codes.add(code)


class TestRoomCreation:
    """Test tạo phòng"""

    def test_create_room_default(self):
        """Tạo phòng với default (15+0)"""
        manager = RoomManager()
        room = manager.create_room()

        assert room is not None
        assert len(room.id) == 6

        room_ref = manager.get_room(room.id)
        assert room_ref is not None
        assert room_ref.status == RoomStatus.WAITING.value
        assert room_ref.clock_white == 900.0
        assert room_ref.clock_black == 900.0

    def test_create_room_custom_time_control(self):
        """Tạo phòng với time control tùy chỉnh"""
        manager = RoomManager()
        room = manager.create_room("5+3")

        room_ref = manager.get_room(room.id)
        assert room_ref.clock_white == 300.0
        assert room_ref.clock_black == 300.0
        assert room_ref.time_control == "5+3"

    def test_parse_time_control(self):
        """Test parse time control"""
        manager = RoomManager()

        assert manager.parse_time_control("15+0") == 900.0
        assert manager.parse_time_control("10+0") == 600.0
        assert manager.parse_time_control("5+3") == 300.0
        assert manager.parse_time_control("invalid") == 900.0  # Default


class TestRoomJoin:
    """Test tham gia phòng"""

    def test_join_empty_room_as_white(self):
        """Tham gia phòng trống làm Trắng"""
        manager = RoomManager()
        room = manager.create_room()

        result = manager.join_room(room.id, "player1")

        assert result['success'] is True
        assert result['color'] == PlayerColor.WHITE
        assert result['game_started'] is False

    def test_join_room_as_black(self):
        """Tham gia phòng làm Đen - game bắt đầu"""
        manager = RoomManager()
        room = manager.create_room()

        # Player 1 joins as white
        manager.join_room(room.id, "player1")

        # Player 2 joins as black
        result = manager.join_room(room.id, "player2")

        assert result['success'] is True
        assert result['color'] == PlayerColor.BLACK
        assert result['game_started'] is True

        # Check room status
        room_ref = manager.get_room(room.id)
        assert room_ref.status == RoomStatus.PLAYING.value

    def test_join_full_room_fails(self):
        """Không thể tham gia phòng đã đầy"""
        manager = RoomManager()
        room = manager.create_room()

        manager.join_room(room.id, "player1")
        manager.join_room(room.id, "player2")

        # Third player tries to join - but room status is PLAYING now
        result = manager.join_room(room.id, "player3")

        assert result['success'] is False
        # Error code is ROOM_PLAYING because the room is now in PLAYING status
        assert result['error']['code'] in ['ROOM_FULL', 'ROOM_PLAYING']

    def test_join_nonexistent_room_fails(self):
        """Không thể tham gia phòng không tồn tại"""
        manager = RoomManager()

        result = manager.join_room("NOTEXIST", "player1")

        assert result['success'] is False
        assert result['error']['code'] == 'ROOM_NOT_FOUND'


class TestPlayerColor:
    """Test lấy màu người chơi"""

    def test_get_player_color_white(self):
        """Lấy màu Trắng"""
        manager = RoomManager()
        room = manager.create_room()
        manager.join_room(room.id, "player1")

        room_ref = manager.get_room(room.id)
        assert room_ref.get_player_color("player1") == PlayerColor.WHITE

    def test_get_player_color_black(self):
        """Lấy màu Đen"""
        manager = RoomManager()
        room = manager.create_room()
        manager.join_room(room.id, "player1")
        manager.join_room(room.id, "player2")

        room_ref = manager.get_room(room.id)
        assert room_ref.get_player_color("player2") == PlayerColor.BLACK

    def test_get_opponent_color(self):
        """Lấy màu đối thủ"""
        manager = RoomManager()
        room = manager.create_room()
        manager.join_room(room.id, "player1")
        manager.join_room(room.id, "player2")

        room_ref = manager.get_room(room.id)
        assert room_ref.get_opponent_color("player1") == PlayerColor.BLACK
        assert room_ref.get_opponent_color("player2") == PlayerColor.WHITE


class TestRoomBySid:
    """Test lấy phòng từ socket ID"""

    def test_set_and_get_player_sid(self):
        """Set và get player SID"""
        manager = RoomManager()
        room = manager.create_room()

        manager.set_player_sid(room.id, "white", "sid1")
        assert manager.get_player_sid(room.id, "white") == "sid1"

        manager.set_player_sid(room.id, "black", "sid2")
        assert manager.get_player_sid(room.id, "black") == "sid2"

    def test_get_room_by_sid(self):
        """Lấy room code từ SID"""
        manager = RoomManager()
        room = manager.create_room()
        manager.join_room(room.id, "player1")

        found_room_code = manager.get_room_by_sid("player1")
        assert found_room_code == room.id


class TestRoomDelete:
    """Test xóa phòng"""

    def test_delete_room(self):
        """Xóa phòng"""
        manager = RoomManager()
        room = manager.create_room()

        assert manager.get_room(room.id) is not None

        manager.delete_room(room.id)

        assert manager.get_room(room.id) is None


class TestRoomIsFull:
    """Test kiểm tra phòng đầy"""

    def test_empty_room_not_full(self):
        """Phòng trống chưa đầy"""
        room = Room("TEST1")
        assert room.is_full() is False

    def test_one_player_not_full(self):
        """Phòng có 1 người chưa đầy"""
        room = Room("TEST2")
        room.white_sid = "player1"
        assert room.is_full() is False

    def test_two_players_full(self):
        """Phòng có 2 người đầy"""
        room = Room("TEST3")
        room.white_sid = "player1"
        room.black_sid = "player2"
        assert room.is_full() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
