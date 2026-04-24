"""Room management - quản lý phòng chơi."""

import uuid
import random
import string
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum


class RoomStatus(str, Enum):
    """Trạng thái phòng"""
    WAITING = "waiting"      # Chờ đối thủ
    PLAYING = "playing"      # Đang chơi
    FINISHED = "finished"    # Đã kết thúc


class PlayerColor(str, Enum):
    """Màu quân cờ"""
    WHITE = "white"
    BLACK = "black"


@dataclass
class Room:
    """Phòng chơi cờ vua."""

    id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    white_player: Optional[str] = None
    black_player: Optional[str] = None
    websocket_white: Optional["WebSocket"] = None
    websocket_black: Optional["WebSocket"] = None
    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves: list = field(default_factory=list)
    clock_times: list = field(default_factory=list)
    status: str = "waiting"  # waiting, playing, finished
    time_control: str = "15+0"

    # Clock tracking (populated by ClockService, kept here for test compatibility)
    clock_white: float = 900.0
    clock_black: float = 900.0
    game_result: Optional[str] = None
    current_turn: str = "white"

    def is_full(self) -> bool:
        """Kiểm tra phòng đã đầy chưa."""
        return self.white_player is not None and self.black_player is not None

    @property
    def waiting_player(self) -> Optional[str]:
        """Trả về player đang chờ."""
        if self.white_player is None:
            return "white"
        elif self.black_player is None:
            return "black"
        return None

    # Aliases for test compatibility (tests use white_sid/black_sid)
    @property
    def white_sid(self) -> Optional[str]:
        return self.white_player

    @white_sid.setter
    def white_sid(self, value: str):
        self.white_player = value

    @property
    def black_sid(self) -> Optional[str]:
        return self.black_player

    @black_sid.setter
    def black_sid(self, value: str):
        self.black_player = value

    def get_opponent_color(self, sid: str) -> Optional[PlayerColor]:
        """Lấy màu của đối thủ"""
        if self.white_player == sid:
            return PlayerColor.BLACK
        elif self.black_player == sid:
            return PlayerColor.WHITE
        return None

    def get_player_color(self, sid: str) -> Optional[PlayerColor]:
        """Lấy màu của người chơi"""
        if self.white_player == sid:
            return PlayerColor.WHITE
        elif self.black_player == sid:
            return PlayerColor.BLACK
        return None


class RoomManager:
    """Quản lý tất cả các phòng."""

    def __init__(self):
        self.rooms: dict = {}
        self.sid_to_room: dict = {}

    def generate_room_code(self) -> str:
        """Tạo mã phòng 6 ký tự alphanumeric"""
        chars = string.ascii_uppercase + string.digits
        while True:
            code = ''.join(random.choices(chars, k=6))
            if code not in self.rooms:
                return code

    def parse_time_control(self, time_control: str) -> float:
        """Parse time control string sang seconds"""
        try:
            minutes = int(time_control.split('+')[0])
            return minutes * 60.0
        except (ValueError, IndexError):
            return 900.0

    def create_room(self, time_control: str = "15+0") -> Room:
        """Tạo phòng mới, trả về Room object."""
        room_code = self.generate_room_code()
        clock_seconds = self.parse_time_control(time_control)
        room = Room(
            id=room_code,
            time_control=time_control,
            clock_white=clock_seconds,
            clock_black=clock_seconds,
            status=RoomStatus.WAITING.value
        )
        self.rooms[room_code] = room
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        """Lấy phòng theo ID."""
        return self.rooms.get(room_id)

    def delete_room(self, room_id: str) -> bool:
        """Xóa phòng."""
        if room_id in self.rooms:
            del self.rooms[room_id]
            return True
        return False

    def join_room(self, room_code: str, sid: str) -> dict:
        """
        Tham gia phòng.
        Returns: dict với success, color, game_started, error
        """
        if room_code not in self.rooms:
            return {
                'success': False,
                'error': {'code': 'ROOM_NOT_FOUND', 'message': 'Phòng không tồn tại'}
            }

        room = self.rooms[room_code]

        if room.status == RoomStatus.PLAYING.value:
            return {
                'success': False,
                'error': {'code': 'ROOM_PLAYING', 'message': 'Ván đấu đang diễn ra'}
            }

        if room.is_full():
            return {
                'success': False,
                'error': {'code': 'ROOM_FULL', 'message': 'Phòng đã đầy'}
            }

        # Gán màu cho người tham gia
        if room.white_player is None:
            room.white_player = sid
            color = PlayerColor.WHITE
        else:
            room.black_player = sid
            color = PlayerColor.BLACK
            room.status = RoomStatus.PLAYING.value

        # Lưu mapping sid -> room
        self.sid_to_room[sid] = room_code

        return {
            'success': True,
            'color': color,
            'game_started': room.status == RoomStatus.PLAYING.value
        }

    def set_player_sid(self, room_code: str, color: str, sid: str):
        """Gán socket ID cho người chơi"""
        room = self.rooms.get(room_code)
        if not room:
            return
        if color == 'white':
            room.white_player = sid
        else:
            room.black_player = sid
        self.sid_to_room[sid] = room_code

    def get_player_sid(self, room_code: str, color: str) -> Optional[str]:
        """Lấy socket ID của người chơi theo màu"""
        room = self.rooms.get(room_code)
        if not room:
            return None
        if color == 'white':
            return room.white_player
        return room.black_player

    def get_room_by_sid(self, sid: str) -> Optional[str]:
        """Lấy mã phòng từ socket ID"""
        return self.sid_to_room.get(sid)

    def get_room_by_sid_obj(self, sid: str) -> Optional[Room]:
        """Lấy Room object từ socket ID"""
        room_code = self.sid_to_room.get(sid)
        if room_code:
            return self.rooms.get(room_code)
        return None

    def on_disconnect(self, sid: str):
        """Xử lý khi client ngắt kết nối"""
        room_code = self.sid_to_room.pop(sid, None)
        if room_code:
            room = self.rooms.get(room_code)
            if room:
                if room.white_player == sid:
                    room.white_player = None
                elif room.black_player == sid:
                    room.black_player = None
                if room.white_player is None and room.black_player is None:
                    del self.rooms[room_code]


# Global room manager
room_manager = RoomManager()
