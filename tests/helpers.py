"""Shared helpers for the test-suite."""

from game.board import Board, square_to_name
from game.rules import generate_legal_moves


def board(fen: str) -> Board:
    return Board.from_fen(fen)


def legal_uci(b: Board, color: str = None) -> set:
    """Set of legal moves (UCI text) for ``color`` (default: side to move)."""
    return {m.uci() for m in generate_legal_moves(b, color or b.turn)}


def targets(b: Board, square: str) -> set:
    """Destination squares (names) of legal moves from ``square``."""
    return {u[2:4] for u in legal_uci(b) if u[:2] == square}


EMPTY_KINGS = "k7/8/8/8/8/8/8/7K"   # placeholder kings tucked away in the corners
