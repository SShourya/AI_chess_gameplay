"""Move representation and the make / undo machinery.

``make_move`` and ``undo_move`` are the only two functions that mutate a
``Board`` during play or search.  ``make_move`` stores everything that cannot
be recomputed (previous castling rights, en-passant square, clocks, hash,
turn) on the ``Move`` object itself, so ``undo_move`` can restore the position
*exactly* -- including the Zobrist hash and repetition history.
"""

from __future__ import annotations

from typing import Optional

from game.board import (
    BLACK_KINGSIDE, BLACK_QUEENSIDE, WHITE_KINGSIDE, WHITE_QUEENSIDE,
    ZOBRIST_CASTLING, ZOBRIST_EN_PASSANT, ZOBRIST_PIECES, ZOBRIST_SIDE,
    Board, Square, name_to_square, square_to_name,
)

__all__ = ["Move", "make_move", "undo_move", "square_to_name", "name_to_square"]

# Moving from *or to* one of these squares removes the listed castling rights
# (king/rook moved, or a rook was captured on its home square).
_CASTLING_MASK = {
    (7, 4): WHITE_KINGSIDE | WHITE_QUEENSIDE,
    (7, 0): WHITE_QUEENSIDE,
    (7, 7): WHITE_KINGSIDE,
    (0, 4): BLACK_KINGSIDE | BLACK_QUEENSIDE,
    (0, 0): BLACK_QUEENSIDE,
    (0, 7): BLACK_KINGSIDE,
}


class Move:
    """A single chess move.

    Public fields: ``from_square``, ``to_square``, ``piece``,
    ``captured_piece``, ``promotion`` (``"q"/"r"/"b"/"n"`` or ``None``),
    ``is_castling`` and ``is_en_passant``.

    The ``prev_*`` fields are filled in by :func:`make_move` and consumed by
    :func:`undo_move`.
    """

    __slots__ = (
        "from_square", "to_square", "piece", "captured_piece", "promotion",
        "is_castling", "is_en_passant",
        "prev_castling", "prev_en_passant", "prev_halfmove", "prev_fullmove",
        "prev_hash", "prev_turn",
    )

    def __init__(
        self,
        from_square: Square,
        to_square: Square,
        piece: str,
        captured_piece: Optional[str] = None,
        promotion: Optional[str] = None,
        is_castling: bool = False,
        is_en_passant: bool = False,
    ) -> None:
        self.from_square = from_square
        self.to_square = to_square
        self.piece = piece
        self.captured_piece = captured_piece
        self.promotion = promotion
        self.is_castling = is_castling
        self.is_en_passant = is_en_passant
        self.prev_castling = 0
        self.prev_en_passant = None
        self.prev_halfmove = 0
        self.prev_fullmove = 1
        self.prev_hash = 0
        self.prev_turn = "w"

    # Two moves are "the same move" if they go between the same squares with
    # the same promotion; the undo bookkeeping is deliberately ignored.
    def key(self):
        return (self.from_square, self.to_square, self.promotion)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Move) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    @property
    def is_capture(self) -> bool:
        return self.captured_piece is not None

    def uci(self) -> str:
        """Long algebraic / UCI style text, e.g. ``"e7e8q"``."""
        return (square_to_name(self.from_square) + square_to_name(self.to_square)
                + (self.promotion or ""))

    def describe(self) -> str:
        """Readable form used in debug output, e.g. ``"e7 -> e5"``."""
        text = f"{square_to_name(self.from_square)} -> {square_to_name(self.to_square)}"
        return text + (f" ={self.promotion.upper()}" if self.promotion else "")

    def __repr__(self) -> str:
        return f"Move({self.uci()})"


def make_move(board: Board, move: Move) -> None:
    """Apply ``move`` to ``board`` in place (including hash and history)."""
    grid = board.grid
    fr, fc = move.from_square
    tr, tc = move.to_square
    piece = move.piece
    color = piece[0]
    ptype = piece[1]
    captured = move.captured_piece

    # -- remember everything undo_move needs --------------------------------
    move.prev_castling = board.castling
    move.prev_en_passant = board.en_passant
    move.prev_halfmove = board.halfmove_clock
    move.prev_fullmove = board.fullmove_number
    move.prev_hash = board.hash
    move.prev_turn = board.turn

    h = board.hash
    zp = ZOBRIST_PIECES
    # Remove old en-passant / castling contributions from the hash.
    if board.en_passant is not None:
        h ^= ZOBRIST_EN_PASSANT[board.en_passant[1]]
    h ^= ZOBRIST_CASTLING[board.castling]

    # -- move the piece -------------------------------------------------------
    grid[fr][fc] = None
    h ^= zp[piece][fr * 8 + fc]

    if move.is_en_passant:
        # The captured pawn is *beside* the origin square, not on the target.
        grid[fr][tc] = None
        h ^= zp[captured][fr * 8 + tc]
    elif captured is not None:
        h ^= zp[captured][tr * 8 + tc]

    placed = color + move.promotion if move.promotion else piece
    grid[tr][tc] = placed
    h ^= zp[placed][tr * 8 + tc]

    if move.is_castling:
        if tc > fc:                      # kingside: rook h -> f
            rook_from, rook_to = 7, 5
        else:                            # queenside: rook a -> d
            rook_from, rook_to = 0, 3
        rook = color + "r"
        grid[fr][rook_from] = None
        grid[fr][rook_to] = rook
        h ^= zp[rook][fr * 8 + rook_from] ^ zp[rook][fr * 8 + rook_to]

    if ptype == "k":
        board.king_pos[color] = (tr, tc)

    # -- castling rights ------------------------------------------------------
    castling = board.castling
    if castling:
        castling &= ~(_CASTLING_MASK.get((fr, fc), 0) | _CASTLING_MASK.get((tr, tc), 0))
    board.castling = castling
    h ^= ZOBRIST_CASTLING[castling]

    # -- en-passant target (only recorded if an enemy pawn could use it) ------
    board.en_passant = None
    if ptype == "p" and abs(tr - fr) == 2:
        enemy_pawn = ("b" if color == "w" else "w") + "p"
        if (tc > 0 and grid[tr][tc - 1] == enemy_pawn) or (tc < 7 and grid[tr][tc + 1] == enemy_pawn):
            board.en_passant = ((fr + tr) // 2, fc)
            h ^= ZOBRIST_EN_PASSANT[fc]

    # -- clocks and turn --------------------------------------------------------
    board.halfmove_clock = 0 if (ptype == "p" or captured is not None) else board.halfmove_clock + 1
    if color == "b":
        board.fullmove_number += 1
    new_turn = "b" if color == "w" else "w"
    if board.turn != new_turn:
        h ^= ZOBRIST_SIDE
    board.turn = new_turn

    board.hash = h
    board.history.append(h)


def undo_move(board: Board, move: Move) -> None:
    """Exactly reverse a previous :func:`make_move` of the same ``move``."""
    grid = board.grid
    fr, fc = move.from_square
    tr, tc = move.to_square
    piece = move.piece
    color = piece[0]

    board.history.pop()

    grid[fr][fc] = piece                 # (un-promoted) piece goes home
    if move.is_en_passant:
        grid[tr][tc] = None
        grid[fr][tc] = move.captured_piece
    else:
        grid[tr][tc] = move.captured_piece

    if move.is_castling:
        if tc > fc:
            rook_from, rook_to = 7, 5
        else:
            rook_from, rook_to = 0, 3
        grid[fr][rook_to] = None
        grid[fr][rook_from] = color + "r"

    if piece[1] == "k":
        board.king_pos[color] = (fr, fc)

    board.castling = move.prev_castling
    board.en_passant = move.prev_en_passant
    board.halfmove_clock = move.prev_halfmove
    board.fullmove_number = move.prev_fullmove
    board.hash = move.prev_hash
    board.turn = move.prev_turn
