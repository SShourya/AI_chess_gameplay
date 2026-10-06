"""High level game state: whose turn it is, game status, captured pieces and
the move log.  This is the only object the UI needs to talk to; it wraps the
board and the rules so the UI never mutates the board directly."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from game.board import Board, Square
from game.move import Move, make_move
from game.pieces import BLACK, PIECE_VALUES, WHITE, color_name, opposite
from game.rules import generate_legal_moves, is_in_check, is_insufficient_material


class GameStatus(Enum):
    PLAYING = "PLAYING"
    CHECK = "CHECK"
    CHECKMATE = "CHECKMATE"
    STALEMATE = "STALEMATE"
    DRAW = "DRAW"


class GameState:
    """A game of chess between a human and the computer."""

    def __init__(self, human_color: str = WHITE) -> None:
        self.human_color = human_color
        self.ai_color = opposite(human_color)
        self.board = Board.initial()
        self.move_log: List[str] = []
        self.captured: Dict[str, List[str]] = {WHITE: [], BLACK: []}   # pieces each side has captured
        self.last_move: Optional[Move] = None
        self.status = GameStatus.PLAYING
        self.winner: Optional[str] = None
        self.draw_reason = ""
        self._legal_cache: List[Move] = []
        self.new_game()

    # ---------------------------------------------------------------- lifecycle
    def new_game(self) -> None:
        """Reset to the starting position."""
        self.board = Board.initial()
        self.move_log = []
        self.captured = {WHITE: [], BLACK: []}
        self.last_move = None
        self.winner = None
        self.draw_reason = ""
        self._refresh_status()

    def load_fen(self, fen: str) -> None:
        """Start from an arbitrary position (used by tests and debugging)."""
        self.board = Board.from_fen(fen)
        self.move_log, self.last_move = [], None
        self.captured = {WHITE: [], BLACK: []}
        self._refresh_status()

    # ------------------------------------------------------------------- queries
    @property
    def turn(self) -> str:
        return self.board.turn

    @property
    def game_over(self) -> bool:
        return self.status in (GameStatus.CHECKMATE, GameStatus.STALEMATE, GameStatus.DRAW)

    @property
    def is_human_turn(self) -> bool:
        return not self.game_over and self.turn == self.human_color

    @property
    def is_ai_turn(self) -> bool:
        return not self.game_over and self.turn == self.ai_color

    @property
    def legal_moves(self) -> List[Move]:
        """Legal moves for the side to move (cached until the next move)."""
        return self._legal_cache

    def legal_moves_from(self, square: Square) -> List[Move]:
        return [m for m in self._legal_cache if m.from_square == square]

    def find_legal_move(self, from_square: Square, to_square: Square,
                        promotion: Optional[str] = None) -> Optional[Move]:
        """The legal move matching the arguments, or ``None`` if illegal."""
        for m in self._legal_cache:
            if m.from_square == from_square and m.to_square == to_square and m.promotion == promotion:
                return m
        return None

    def needs_promotion_choice(self, from_square: Square, to_square: Square) -> bool:
        """True if moving from -> to is a pawn promotion (the UI must ask which piece)."""
        return any(m.from_square == from_square and m.to_square == to_square and m.promotion
                   for m in self._legal_cache)

    def material_balance(self) -> int:
        """Material captured by White minus material captured by Black (pawns)."""
        won_by_white = sum(PIECE_VALUES[p[1]] for p in self.captured[WHITE])
        won_by_black = sum(PIECE_VALUES[p[1]] for p in self.captured[BLACK])
        return (won_by_white - won_by_black) // 100

    # ------------------------------------------------------------------- actions
    def play(self, from_square: Square, to_square: Square, promotion: Optional[str] = None) -> bool:
        """Play a move given by squares.  Returns ``False`` (and changes
        nothing) if it is not legal -- this is the single gate through which
        every move, human or AI, must pass."""
        if self.game_over:
            return False
        move = self.find_legal_move(from_square, to_square, promotion)
        if move is None:
            return False
        mover = move.piece[0]
        make_move(self.board, move)
        if move.captured_piece:
            self.captured[mover].append(move.captured_piece)
        self.last_move = move
        self.move_log.append(move.uci())
        self._refresh_status()
        return True

    # ------------------------------------------------------------------ internals
    def _refresh_status(self) -> None:
        """Recompute legal moves and the game status for the side to move."""
        board = self.board
        side = board.turn
        self._legal_cache = generate_legal_moves(board, side)
        in_check = is_in_check(board, side)
        self.winner = None
        self.draw_reason = ""

        if not self._legal_cache:
            if in_check:
                self.status = GameStatus.CHECKMATE
                self.winner = opposite(side)
            else:
                self.status = GameStatus.STALEMATE
                self.draw_reason = "Stalemate: no legal moves but not in check"
        elif is_insufficient_material(board):
            self.status = GameStatus.DRAW
            self.draw_reason = "Insufficient material"
        elif board.halfmove_clock >= 100:
            self.status = GameStatus.DRAW
            self.draw_reason = "Fifty-move rule"
        elif board.repetition_count() >= 3:
            self.status = GameStatus.DRAW
            self.draw_reason = "Threefold repetition"
        elif in_check:
            self.status = GameStatus.CHECK
        else:
            self.status = GameStatus.PLAYING

    def king_in_check_square(self) -> Optional[Square]:
        """Square of the king to highlight when the side to move is in check."""
        if self.status in (GameStatus.CHECK, GameStatus.CHECKMATE):
            return self.board.king_pos[self.turn]
        return None

    def status_text(self) -> str:
        if self.status == GameStatus.CHECKMATE:
            return f"Checkmate! {color_name(self.winner)} wins."
        if self.status == GameStatus.STALEMATE:
            return "Stalemate - draw."
        if self.status == GameStatus.DRAW:
            return f"Draw - {self.draw_reason.lower()}."
        if self.status == GameStatus.CHECK:
            return f"{color_name(self.turn)} is in check!"
        return "Game in progress"
