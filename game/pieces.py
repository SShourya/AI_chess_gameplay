"""Piece constants and lookup tables shared by the whole engine.

Pieces are two-character strings: colour + type, e.g. ``"wk"`` (white king)
or ``"bp"`` (black pawn).  Strings are used (instead of objects) because they
are immutable, hashable, cheap to compare and trivially printable, which
matters a lot inside a pure-Python search loop.
"""

from __future__ import annotations

from typing import Optional

WHITE = "w"
BLACK = "b"

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = "p", "n", "b", "r", "q", "k"
PIECE_TYPES = (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
PROMOTION_TYPES = (QUEEN, ROOK, BISHOP, KNIGHT)

#: Every valid piece code (used to build the Zobrist tables).
ALL_PIECES = tuple(c + t for c in (WHITE, BLACK) for t in PIECE_TYPES)

#: Classical material values in centipawns.
PIECE_VALUES = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 20000}

#: Unicode "solid" glyphs (U+265A..U+265F).  We draw *both* colours with the
#: solid glyph and tint them, which gives much better contrast than the hollow
#: white glyphs.
UNICODE_SOLID = {
    KING: "\u265A", QUEEN: "\u265B", ROOK: "\u265C",
    BISHOP: "\u265D", KNIGHT: "\u265E", PAWN: "\u265F",
}

PIECE_NAMES = {
    PAWN: "Pawn", KNIGHT: "Knight", BISHOP: "Bishop",
    ROOK: "Rook", QUEEN: "Queen", KING: "King",
}

# Movement vectors as (d_row, d_col).  Row 0 is rank 8 (black's back rank).
KNIGHT_OFFSETS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
KING_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
BISHOP_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ROOK_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
QUEEN_DIRECTIONS = BISHOP_DIRECTIONS + ROOK_DIRECTIONS


def opposite(color: str) -> str:
    """Return the other colour."""
    return BLACK if color == WHITE else WHITE


def color_of(piece: Optional[str]) -> Optional[str]:
    return piece[0] if piece else None


def type_of(piece: Optional[str]) -> Optional[str]:
    return piece[1] if piece else None


def piece_name(piece: str) -> str:
    """Human readable name, e.g. ``"White Knight"``."""
    colour = "White" if piece[0] == WHITE else "Black"
    return f"{colour} {PIECE_NAMES[piece[1]]}"


def color_name(color: str) -> str:
    return "White" if color == WHITE else "Black"
