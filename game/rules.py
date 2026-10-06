"""Chess rules: attack detection and move generation.

Two-stage move generation
-------------------------
1. :func:`generate_pseudo_legal_moves` produces every move that obeys the
   *movement* rules of each piece (plus castling and en passant) but may leave
   the mover's own king in check.
2. :func:`generate_legal_moves` plays every pseudo-legal move on the board,
   asks :func:`is_in_check` whether the mover's king is exposed, undoes the
   move, and keeps only the safe ones.

The AI and the UI only ever use stage 2, so an illegal move can never be
played.
"""

from __future__ import annotations

from typing import List

from game.board import (
    BLACK_KINGSIDE, BLACK_QUEENSIDE, WHITE_KINGSIDE, WHITE_QUEENSIDE,
    Board, Square,
)
from game.move import Move, make_move, undo_move
from game.pieces import (
    BISHOP_DIRECTIONS, KING_OFFSETS, KNIGHT_OFFSETS, PROMOTION_TYPES,
    QUEEN_DIRECTIONS, ROOK_DIRECTIONS, opposite,
)


# ---------------------------------------------------------------------------
# Attack detection
# ---------------------------------------------------------------------------
def is_square_attacked(board: Board, square: Square, attacking_color: str) -> bool:
    """Is ``square`` attacked by any piece of ``attacking_color``?

    Instead of generating the attacker's moves we look *outwards* from the
    square: for pawns/knights/kings we test the few squares from which such a
    piece could hit it, for sliders we walk each ray until the first piece.
    This is what check detection, castling validation and legal-move
    filtering are built on.
    """
    grid = board.grid
    r, c = square

    # Pawns attack diagonally forward, so the attacker sits diagonally *behind*
    # the target from the attacker's point of view.
    if attacking_color == "w":
        pr = r + 1                       # white pawns move towards row 0
        if pr < 8:
            if c > 0 and grid[pr][c - 1] == "wp":
                return True
            if c < 7 and grid[pr][c + 1] == "wp":
                return True
    else:
        pr = r - 1
        if pr >= 0:
            if c > 0 and grid[pr][c - 1] == "bp":
                return True
            if c < 7 and grid[pr][c + 1] == "bp":
                return True

    knight = attacking_color + "n"
    for dr, dc in KNIGHT_OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 8 and 0 <= cc < 8 and grid[rr][cc] == knight:
            return True

    king = attacking_color + "k"
    for dr, dc in KING_OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 8 and 0 <= cc < 8 and grid[rr][cc] == king:
            return True

    bishop, rook, queen = attacking_color + "b", attacking_color + "r", attacking_color + "q"
    for dr, dc in BISHOP_DIRECTIONS:
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            p = grid[rr][cc]
            if p is not None:
                if p == bishop or p == queen:
                    return True
                break
            rr += dr
            cc += dc
    for dr, dc in ROOK_DIRECTIONS:
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            p = grid[rr][cc]
            if p is not None:
                if p == rook or p == queen:
                    return True
                break
            rr += dr
            cc += dc
    return False


def is_in_check(board: Board, color: str) -> bool:
    """Is ``color``'s king currently attacked?"""
    king_square = board.king_pos[color]
    if king_square is None:              # only possible in artificial test positions
        return False
    return is_square_attacked(board, king_square, opposite(color))


# ---------------------------------------------------------------------------
# Pseudo-legal generation
# ---------------------------------------------------------------------------
def _add_pawn_move(moves: List[Move], frm: Square, to: Square, piece: str,
                   captured, promo_row: int, en_passant: bool = False) -> None:
    if to[0] == promo_row:
        for promo in PROMOTION_TYPES:
            moves.append(Move(frm, to, piece, captured, promotion=promo))
    else:
        moves.append(Move(frm, to, piece, captured, is_en_passant=en_passant))


def _gen_pawn_moves(board: Board, r: int, c: int, color: str, moves: List[Move]) -> None:
    grid = board.grid
    piece = color + "p"
    direction = -1 if color == "w" else 1
    start_row = 6 if color == "w" else 1
    promo_row = 0 if color == "w" else 7
    nr = r + direction
    if not 0 <= nr < 8:
        return

    # Forward pushes
    if grid[nr][c] is None:
        _add_pawn_move(moves, (r, c), (nr, c), piece, None, promo_row)
        if r == start_row and grid[nr + direction][c] is None:
            moves.append(Move((r, c), (nr + direction, c), piece))

    # Captures (including en passant)
    for dc in (-1, 1):
        nc = c + dc
        if not 0 <= nc < 8:
            continue
        target = grid[nr][nc]
        if target is not None:
            if target[0] != color:
                _add_pawn_move(moves, (r, c), (nr, nc), piece, target, promo_row)
        elif board.en_passant == (nr, nc) and board.turn == color:
            victim = grid[r][nc]
            if victim == opposite(color) + "p":
                moves.append(Move((r, c), (nr, nc), piece, victim, is_en_passant=True))


def _gen_step_moves(board: Board, r: int, c: int, color: str, offsets, moves: List[Move]) -> None:
    grid = board.grid
    piece = grid[r][c]
    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 8 and 0 <= cc < 8:
            target = grid[rr][cc]
            if target is None:
                moves.append(Move((r, c), (rr, cc), piece))
            elif target[0] != color:
                moves.append(Move((r, c), (rr, cc), piece, target))


def _gen_sliding_moves(board: Board, r: int, c: int, color: str, directions, moves: List[Move]) -> None:
    grid = board.grid
    piece = grid[r][c]
    for dr, dc in directions:
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            target = grid[rr][cc]
            if target is None:
                moves.append(Move((r, c), (rr, cc), piece))
            else:
                if target[0] != color:
                    moves.append(Move((r, c), (rr, cc), piece, target))
                break
            rr += dr
            cc += dc


def _gen_castling_moves(board: Board, color: str, moves: List[Move]) -> None:
    """Castling, fully validated here (rights, empty squares, no check on the
    king's start, transit or destination square)."""
    row = 7 if color == "w" else 0
    grid = board.grid
    king = color + "k"
    if grid[row][4] != king:
        return
    kingside_bit = WHITE_KINGSIDE if color == "w" else BLACK_KINGSIDE
    queenside_bit = WHITE_QUEENSIDE if color == "w" else BLACK_QUEENSIDE
    if not (board.castling & (kingside_bit | queenside_bit)):
        return
    enemy = opposite(color)
    if is_square_attacked(board, (row, 4), enemy):
        return                                     # cannot castle out of check

    rook = color + "r"
    if (board.castling & kingside_bit) and grid[row][7] == rook \
            and grid[row][5] is None and grid[row][6] is None \
            and not is_square_attacked(board, (row, 5), enemy) \
            and not is_square_attacked(board, (row, 6), enemy):
        moves.append(Move((row, 4), (row, 6), king, is_castling=True))

    if (board.castling & queenside_bit) and grid[row][0] == rook \
            and grid[row][1] is None and grid[row][2] is None and grid[row][3] is None \
            and not is_square_attacked(board, (row, 3), enemy) \
            and not is_square_attacked(board, (row, 2), enemy):
        moves.append(Move((row, 4), (row, 2), king, is_castling=True))


def generate_pseudo_legal_moves(board: Board, color: str) -> List[Move]:
    """All moves ``color`` could make if leaving its king in check were allowed.

    Castling is the one exception: it is only generated when its own
    "not through check" conditions hold, because those cannot be checked by
    the make-and-test filter in :func:`generate_legal_moves`.
    """
    moves: List[Move] = []
    grid = board.grid
    for r in range(8):
        row = grid[r]
        for c in range(8):
            piece = row[c]
            if piece is None or piece[0] != color:
                continue
            ptype = piece[1]
            if ptype == "p":
                _gen_pawn_moves(board, r, c, color, moves)
            elif ptype == "n":
                _gen_step_moves(board, r, c, color, KNIGHT_OFFSETS, moves)
            elif ptype == "b":
                _gen_sliding_moves(board, r, c, color, BISHOP_DIRECTIONS, moves)
            elif ptype == "r":
                _gen_sliding_moves(board, r, c, color, ROOK_DIRECTIONS, moves)
            elif ptype == "q":
                _gen_sliding_moves(board, r, c, color, QUEEN_DIRECTIONS, moves)
            else:  # king
                _gen_step_moves(board, r, c, color, KING_OFFSETS, moves)
    _gen_castling_moves(board, color, moves)
    return moves


def generate_legal_moves(board: Board, color: str) -> List[Move]:
    """Legal moves for ``color``: pseudo-legal moves that do not leave the
    mover's own king attacked.  Every candidate is played, tested and undone,
    so the board is unchanged when this function returns."""
    legal: List[Move] = []
    for move in generate_pseudo_legal_moves(board, color):
        make_move(board, move)
        if not is_in_check(board, color):
            legal.append(move)
        undo_move(board, move)
    return legal


# ---------------------------------------------------------------------------
# Game-end helpers
# ---------------------------------------------------------------------------
def is_checkmate(board: Board, color: str) -> bool:
    return is_in_check(board, color) and not generate_legal_moves(board, color)


def is_stalemate(board: Board, color: str) -> bool:
    return not is_in_check(board, color) and not generate_legal_moves(board, color)


def is_insufficient_material(board: Board) -> bool:
    """True when neither side can possibly deliver checkmate.

    Covers K v K, K+B v K, K+N v K and any number of bishops that all stand on
    squares of one colour (with no other pieces except kings)."""
    knights = 0
    bishop_square_colors = set()
    for r in range(8):
        for c in range(8):
            piece = board.grid[r][c]
            if piece is None or piece[1] == "k":
                continue
            t = piece[1]
            if t in "pqr":
                return False
            if t == "n":
                knights += 1
            else:
                bishop_square_colors.add((r + c) % 2)
    if knights == 0:
        return len(bishop_square_colors) <= 1
    return knights == 1 and not bishop_square_colors


def perft(board: Board, depth: int) -> int:
    """Count leaf nodes of the legal move tree; the standard way to verify a
    move generator against published reference numbers."""
    if depth == 0:
        return 1
    total = 0
    for move in generate_legal_moves(board, board.turn):
        if depth == 1:
            total += 1
            continue
        make_move(board, move)
        total += perft(board, depth - 1)
        undo_move(board, move)
    return total
