"""Board representation.

The board is an 8x8 list of lists (``board.grid[row][col]``).  Row 0 is
rank 8 (Black's back rank) and row 7 is rank 1, so the printed grid looks
exactly like a chess diagram from White's point of view::

    grid[0] = ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"]   # rank 8
    ...
    grid[7] = ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"]   # rank 1

Besides the piece placement the board tracks everything else that defines a
chess position: side to move, castling rights (bitmask), en-passant target,
the half-move clock (fifty-move rule) and the move number.

Position hashing
----------------
``board.hash`` is a 64-bit **Zobrist hash**: one random 64-bit number per
(piece, square) pair, plus numbers for the side to move, each castling-rights
combination and each en-passant file.  The hash of a position is the XOR of
the numbers for everything that is true in it.  Because XOR is its own
inverse, ``make_move`` / ``undo_move`` update the hash *incrementally* in a
handful of XORs instead of rebuilding a key from all 64 squares.  The
transposition table and the repetition detector both use this integer as their
hashable position key.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from game.pieces import ALL_PIECES, BLACK, WHITE, opposite

Square = Tuple[int, int]

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Castling-rights bit flags.
WHITE_KINGSIDE = 1
WHITE_QUEENSIDE = 2
BLACK_KINGSIDE = 4
BLACK_QUEENSIDE = 8

_CASTLING_FEN_BITS = (
    ("K", WHITE_KINGSIDE), ("Q", WHITE_QUEENSIDE),
    ("k", BLACK_KINGSIDE), ("q", BLACK_QUEENSIDE),
)

# ---------------------------------------------------------------------------
# Zobrist tables (deterministic seed => reproducible hashes between runs)
# ---------------------------------------------------------------------------
_rng = random.Random(0xC4E55)
ZOBRIST_PIECES: Dict[str, List[int]] = {
    piece: [_rng.getrandbits(64) for _ in range(64)] for piece in ALL_PIECES
}
ZOBRIST_SIDE = _rng.getrandbits(64)          # XORed in when Black is to move
ZOBRIST_CASTLING = [_rng.getrandbits(64) for _ in range(16)]
ZOBRIST_EN_PASSANT = [_rng.getrandbits(64) for _ in range(8)]


class Board:
    """A complete chess position plus the history needed for repetition."""

    def __init__(
        self,
        grid: List[List[Optional[str]]],
        turn: str = WHITE,
        castling: int = 0,
        en_passant: Optional[Square] = None,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
    ) -> None:
        self.grid = grid
        self.turn = turn
        self.castling = castling
        self.en_passant = en_passant
        self.halfmove_clock = halfmove_clock
        self.fullmove_number = fullmove_number

        self.king_pos: Dict[str, Optional[Square]] = {WHITE: None, BLACK: None}
        self._locate_kings()

        self.hash = self.compute_hash()
        #: Hash of every position reached so far (including the current one).
        self.history: List[int] = [self.hash]

    # ------------------------------------------------------------------ setup
    @classmethod
    def initial(cls) -> "Board":
        """The standard starting position."""
        return cls.from_fen(START_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        """Build a board from a FEN string (the last two fields are optional)."""
        parts = fen.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid FEN (need at least 4 fields): {fen!r}")
        rows = parts[0].split("/")
        if len(rows) != 8:
            raise ValueError(f"Invalid FEN piece placement: {parts[0]!r}")

        grid: List[List[Optional[str]]] = []
        for row in rows:
            cells: List[Optional[str]] = []
            for ch in row:
                if ch.isdigit():
                    cells.extend([None] * int(ch))
                elif ch.lower() in "pnbrqk":
                    cells.append(("w" if ch.isupper() else "b") + ch.lower())
                else:
                    raise ValueError(f"Invalid FEN character {ch!r}")
            if len(cells) != 8:
                raise ValueError(f"FEN row does not have 8 squares: {row!r}")
            grid.append(cells)

        turn = WHITE if parts[1] == "w" else BLACK
        castling = 0
        for letter, bit in _CASTLING_FEN_BITS:
            if letter in parts[2]:
                castling |= bit
        en_passant = None if parts[3] == "-" else _name_to_square(parts[3])
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        fullmove = int(parts[5]) if len(parts) > 5 else 1
        return cls(grid, turn, castling, en_passant, halfmove, fullmove)

    def to_fen(self) -> str:
        """Serialise the position to FEN."""
        rows = []
        for row in self.grid:
            out, empty = "", 0
            for cell in row:
                if cell is None:
                    empty += 1
                    continue
                if empty:
                    out += str(empty)
                    empty = 0
                out += cell[1].upper() if cell[0] == WHITE else cell[1]
            if empty:
                out += str(empty)
            rows.append(out)
        castling = "".join(l for l, bit in _CASTLING_FEN_BITS if self.castling & bit) or "-"
        ep = "-" if self.en_passant is None else _square_to_name(self.en_passant)
        return f"{'/'.join(rows)} {self.turn} {castling} {ep} {self.halfmove_clock} {self.fullmove_number}"

    def copy(self) -> "Board":
        """Independent deep copy (used to hand a position to the AI thread)."""
        clone = Board.__new__(Board)
        clone.grid = [row[:] for row in self.grid]
        clone.turn = self.turn
        clone.castling = self.castling
        clone.en_passant = self.en_passant
        clone.halfmove_clock = self.halfmove_clock
        clone.fullmove_number = self.fullmove_number
        clone.king_pos = dict(self.king_pos)
        clone.hash = self.hash
        clone.history = self.history[:]
        return clone

    # ---------------------------------------------------------------- helpers
    def _locate_kings(self) -> None:
        for r in range(8):
            for c in range(8):
                piece = self.grid[r][c]
                if piece == "wk":
                    self.king_pos[WHITE] = (r, c)
                elif piece == "bk":
                    self.king_pos[BLACK] = (r, c)

    def piece_at(self, square: Square) -> Optional[str]:
        return self.grid[square[0]][square[1]]

    def compute_hash(self) -> int:
        """Recompute the Zobrist hash from scratch (used at setup and in tests
        to verify the incremental updates done by ``make_move``)."""
        h = 0
        for r in range(8):
            for c in range(8):
                piece = self.grid[r][c]
                if piece:
                    h ^= ZOBRIST_PIECES[piece][r * 8 + c]
        if self.turn == BLACK:
            h ^= ZOBRIST_SIDE
        h ^= ZOBRIST_CASTLING[self.castling]
        if self.en_passant is not None:
            h ^= ZOBRIST_EN_PASSANT[self.en_passant[1]]
        return h

    def repetition_count(self) -> int:
        """How many times the current position has occurred in this game."""
        return self.history.count(self.hash)

    def pieces_of(self, color: str):
        """Yield ``(piece, row, col)`` for every piece of ``color``."""
        for r in range(8):
            for c in range(8):
                piece = self.grid[r][c]
                if piece and piece[0] == color:
                    yield piece, r, c

    def __str__(self) -> str:
        lines = []
        for r in range(8):
            cells = [p if p else ".." for p in self.grid[r]]
            lines.append(f"{8 - r}  " + " ".join(cells))
        lines.append("   " + "  ".join("abcdefgh"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Square <-> algebraic name helpers (also re-exported by game.move)
# ---------------------------------------------------------------------------
def _square_to_name(square: Square) -> str:
    return "abcdefgh"[square[1]] + str(8 - square[0])


def _name_to_square(name: str) -> Square:
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise ValueError(f"Invalid square name: {name!r}")
    return 8 - int(name[1]), "abcdefgh".index(name[0])


square_to_name = _square_to_name
name_to_square = _name_to_square
