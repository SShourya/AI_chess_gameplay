"""Static position evaluation.

``evaluate_board(board, perspective)`` returns a score in centipawns
(100 = one pawn) **from the point of view of ``perspective``**: positive means
that colour is better.  With the default ``perspective="w"`` this is the
classic "White score - Black score".  The search passes the AI's own colour, so
the AI's nodes are *maximising* and the human's nodes are *minimising*.

Terms
-----
* material                      (pawn 100 .. queen 900, king 20000)
* piece-square tables           (Michniewski's "simplified evaluation")
* centre control                (pieces on d4/e4/d5/e5)
* piece development             (minor pieces still on their home squares)
* king safety                   (pawn shield in the middlegame)
* castling                      (bonus once castled, penalty for losing the right)
* pawn structure                (doubled, isolated and passed pawns)
* mobility                      (squares reachable by N, B, R, Q)
* check                         (small penalty for the side in check)
* endgame "mop-up"              (drive the losing king to the edge)
* checkmate / stalemate         (see :func:`terminal_score`)
"""

from __future__ import annotations

from typing import Optional

from game.board import BLACK_KINGSIDE, BLACK_QUEENSIDE, WHITE_KINGSIDE, WHITE_QUEENSIDE, Board
from game.pieces import BISHOP_DIRECTIONS, KNIGHT_OFFSETS, PIECE_VALUES, QUEEN_DIRECTIONS, ROOK_DIRECTIONS
from game.rules import generate_legal_moves, is_in_check

CHECKMATE_SCORE = 1_000_000

# ---------------------------------------------------------------------------
# Piece-square tables, written from White's point of view with row 0 = rank 8
# (exactly how the board grid is stored).  Black uses the vertically mirrored
# table.  Positive numbers = squares the piece likes.
# ---------------------------------------------------------------------------
_PAWN = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
_KNIGHT = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]
_BISHOP = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]
_ROOK = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0],
]
_QUEEN = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20],
]
_KING_MIDDLEGAME = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20],
]
_KING_ENDGAME = [
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
]


def _mirror(table):
    return [row[:] for row in reversed(table)]


_BASE_TABLES = {"p": _PAWN, "n": _KNIGHT, "b": _BISHOP, "r": _ROOK, "q": _QUEEN}
#: PST[piece][row][col] for every piece code (kings handled separately).
PIECE_SQUARE_TABLES = {}
for _t, _tbl in _BASE_TABLES.items():
    PIECE_SQUARE_TABLES["w" + _t] = _tbl
    PIECE_SQUARE_TABLES["b" + _t] = _mirror(_tbl)
KING_TABLES = {
    ("w", False): _KING_MIDDLEGAME, ("b", False): _mirror(_KING_MIDDLEGAME),
    ("w", True): _KING_ENDGAME, ("b", True): _mirror(_KING_ENDGAME),
}

# ---------------------------------------------------------------------------
# Tunable weights (centipawns)
# ---------------------------------------------------------------------------
CENTER_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}
CENTER_BONUS = 10
UNDEVELOPED_PENALTY = 12
DOUBLED_PAWN_PENALTY = 15
ISOLATED_PAWN_PENALTY = 15
PASSED_PAWN_BONUS = (0, 8, 15, 30, 55, 90, 140)     # indexed by squares advanced
SHIELD_NEAR_BONUS = 12       # friendly pawn directly in front of the king
SHIELD_FAR_BONUS = 6         # friendly pawn two squares in front
OPEN_KING_FILE_PENALTY = 15
CASTLED_BONUS = 35
LOST_CASTLING_PENALTY = 20
IN_CHECK_PENALTY = 25
MOBILITY_WEIGHT = {"n": 4, "b": 4, "r": 2, "q": 1}

_HOME_MINOR_SQUARES = {
    "wn": {(7, 1), (7, 6)}, "wb": {(7, 2), (7, 5)},
    "bn": {(0, 1), (0, 6)}, "bb": {(0, 2), (0, 5)},
}
_TYPE_INDEX = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def _is_endgame(counts) -> bool:
    """Endgame if there are no queens, or every side that has a queen has no
    rook and at most one minor piece."""
    for side in (0, 1):
        if counts[side][4] > 0 and (counts[side][3] > 0 or counts[side][1] + counts[side][2] > 1):
            return False
    return True


def _mobility(grid, r, c, ptype, color) -> int:
    """Number of squares a knight/bishop/rook/queen can move to or capture on."""
    count = 0
    if ptype == "n":
        for dr, dc in KNIGHT_OFFSETS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                t = grid[rr][cc]
                if t is None or t[0] != color:
                    count += 1
        return count
    directions = BISHOP_DIRECTIONS if ptype == "b" else ROOK_DIRECTIONS if ptype == "r" else QUEEN_DIRECTIONS
    for dr, dc in directions:
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            t = grid[rr][cc]
            if t is None:
                count += 1
            else:
                if t[0] != color:
                    count += 1
                break
            rr += dr
            cc += dc
    return count


def terminal_score(board: Board, ai_color: str, ply: int) -> Optional[float]:
    """Score for a position with no legal moves, else ``None``.

    Checkmate is worth ``+/-CHECKMATE_SCORE`` *minus the distance from the
    root* (``ply``), so the AI prefers the fastest mate and, when losing, the
    slowest one.  Stalemate is a draw (0)."""
    if generate_legal_moves(board, board.turn):
        return None
    if is_in_check(board, board.turn):
        mate = CHECKMATE_SCORE - ply
        return -mate if board.turn == ai_color else mate
    return 0


def evaluate_board(board: Board, perspective: str = "w", detect_terminal: bool = False) -> float:
    """Static evaluation in centipawns from ``perspective``'s point of view.

    With ``detect_terminal=True`` checkmate/stalemate are detected first (this
    generates legal moves so it is slower; the search does its own terminal
    detection and leaves this off)."""
    if detect_terminal:
        terminal = terminal_score(board, perspective, 0)
        if terminal is not None:
            return terminal

    grid = board.grid
    pieces = []
    counts = [[0] * 6, [0] * 6]                     # per side: p n b r q k
    for r in range(8):
        row = grid[r]
        for c in range(8):
            p = row[c]
            if p is not None:
                side = 0 if p[0] == "w" else 1
                counts[side][_TYPE_INDEX[p[1]]] += 1
                pieces.append((p, r, c, side))

    # Dead position (K v K, K+minor v K): a draw whatever the "material" says.
    if all(counts[s][0] + counts[s][3] + counts[s][4] == 0 for s in (0, 1)) \
            and counts[0][1] + counts[0][2] + counts[1][1] + counts[1][2] <= 1:
        return 0

    endgame = _is_endgame(counts)
    totals = [0, 0]                                 # [white, black]
    pawn_rows = [[[] for _ in range(8)], [[] for _ in range(8)]]
    king_square = [None, None]

    for piece, r, c, side in pieces:
        color, ptype = piece[0], piece[1]
        value = PIECE_VALUES[ptype]
        if ptype == "k":
            king_square[side] = (r, c)
            totals[side] += KING_TABLES[(color, endgame)][r][c]
            continue
        totals[side] += value + PIECE_SQUARE_TABLES[piece][r][c]

        if ptype == "p":
            pawn_rows[side][c].append(r)
        else:
            if ptype in MOBILITY_WEIGHT:
                totals[side] += MOBILITY_WEIGHT[ptype] * _mobility(grid, r, c, ptype, color)
            if not endgame and (r, c) in _HOME_MINOR_SQUARES.get(piece, ()):
                totals[side] -= UNDEVELOPED_PENALTY        # development
        if (r, c) in CENTER_SQUARES and ptype in "pnb":
            totals[side] += CENTER_BONUS                   # centre control

    # -- pawn structure --------------------------------------------------------
    for side in (0, 1):
        mine, theirs = pawn_rows[side], pawn_rows[1 - side]
        for f in range(8):
            rows = mine[f]
            if not rows:
                continue
            if len(rows) > 1:
                totals[side] -= DOUBLED_PAWN_PENALTY * (len(rows) - 1)
            if (f == 0 or not mine[f - 1]) and (f == 7 or not mine[f + 1]):
                totals[side] -= ISOLATED_PAWN_PENALTY * len(rows)
            for r in rows:
                passed = True
                for ef in (f - 1, f, f + 1):
                    if 0 <= ef < 8:
                        for er in theirs[ef]:
                            if (er < r) if side == 0 else (er > r):
                                passed = False
                                break
                    if not passed:
                        break
                if passed:
                    advanced = (6 - r) if side == 0 else (r - 1)
                    totals[side] += PASSED_PAWN_BONUS[max(0, min(6, advanced))] * (2 if endgame else 1)

    # -- king safety and castling (middlegame only) ------------------------------
    if not endgame:
        for side in (0, 1):
            ksq = king_square[side]
            if ksq is None:
                continue
            kr, kc = ksq
            back_rank = 7 if side == 0 else 0
            step = -1 if side == 0 else 1
            if kr == back_rank:
                for f in range(max(0, kc - 1), min(7, kc + 1) + 1):
                    rows = pawn_rows[side][f]
                    if (kr + step) in rows:
                        totals[side] += SHIELD_NEAR_BONUS
                    elif (kr + 2 * step) in rows:
                        totals[side] += SHIELD_FAR_BONUS
                    elif not rows:
                        totals[side] -= OPEN_KING_FILE_PENALTY
                if kc in (6, 2):
                    totals[side] += CASTLED_BONUS
                elif kc == 4:
                    rights = (WHITE_KINGSIDE | WHITE_QUEENSIDE) if side == 0 else (BLACK_KINGSIDE | BLACK_QUEENSIDE)
                    if not board.castling & rights:
                        totals[side] -= LOST_CASTLING_PENALTY

    # -- check ------------------------------------------------------------------
    if is_in_check(board, board.turn):
        totals[0 if board.turn == "w" else 1] -= IN_CHECK_PENALTY

    # -- endgame mop-up: drive the lone king to the edge ------------------------
    if endgame and king_square[0] and king_square[1]:
        def army(side):
            n = counts[side]
            return n[1] * 320 + n[2] * 330 + n[3] * 500 + n[4] * 900
        lead = army(0) - army(1)
        if abs(lead) >= 300:
            winner = 0 if lead > 0 else 1
            lr, lc = king_square[1 - winner]
            wr, wc = king_square[winner]
            edge_distance = abs(lr - 3.5) + abs(lc - 3.5)            # 1 (centre) .. 7 (corner)
            king_gap = abs(lr - wr) + abs(lc - wc)                    # 2 .. 14
            totals[winner] += int(10 * edge_distance + 4 * (14 - king_gap))

    score = totals[0] - totals[1]
    return score if perspective == "w" else -score
