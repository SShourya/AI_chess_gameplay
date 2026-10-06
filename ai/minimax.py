"""Minimax search with alpha-beta pruning.

Perspective
-----------
Scores are always expressed **from the AI's point of view** (see
``SearchContext.ai_color``): positive = good for the AI.  Therefore

* a node where the AI is to move is a *maximising* node,
* a node where the human is to move is a *minimising* node.

Both :func:`minimax` and :func:`alpha_beta` search exactly the same tree and
return exactly the same value; alpha-beta simply skips branches that cannot
change the result.

Search details shared by both algorithms
----------------------------------------
* No legal moves  -> checkmate (``+/-(CHECKMATE_SCORE - ply)``) or stalemate (0).
* Fifty-move rule and repetition inside the tree score 0 (draw).
* *Check extension*: a leaf where the side to move is in check is searched one
  ply deeper, so forced mates are not missed at the horizon.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ai.evaluation import CHECKMATE_SCORE, evaluate_board
from ai.transposition import EXACT, LOWER, UPPER, TranspositionTable
from game.board import Board
from game.move import Move, make_move, undo_move
from game.pieces import PIECE_VALUES
from game.rules import generate_legal_moves, is_in_check

#: Set to ``True`` (or run ``python main.py --debug``) to print search details.
DEBUG = False

#: Default depth in plies; can be overridden by the difficulty setting.
SEARCH_DEPTH = 3
DIFFICULTY_DEPTHS = {"Easy": 2, "Medium": 3, "Hard": 4}

INFINITY = float("inf")
MATE_THRESHOLD = CHECKMATE_SCORE - 1000
CHECK_EXTENSION_PLIES = 4          # how far past the nominal depth checks may extend

# Move-ordering bonuses (bigger = searched earlier).
MATE_BONUS = 10_000_000
TT_BONUS = 9_000_000
CAPTURE_BONUS = 1_000_000
PROMOTION_BONUS = 900_000
CHECK_BONUS = 500_000
_RANK = {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}


class SearchCancelled(Exception):
    """Raised inside the search when the UI asks it to stop."""


@dataclass
class SearchStats:
    depth: int = 0
    nodes: int = 0            # positions visited
    cutoffs: int = 0          # times alpha >= beta stopped a node early
    pruned: int = 0           # sibling moves skipped because of those cutoffs
    tt_hits: int = 0
    root_moves: int = 0       # legal moves generated at the root
    best_score: float = 0.0
    elapsed: float = 0.0


@dataclass
class SearchContext:
    ai_color: str = "b"
    stats: SearchStats = field(default_factory=SearchStats)
    tt: Optional[TranspositionTable] = None
    use_ordering: bool = True
    max_ply: int = 64
    cancel_event: Optional[threading.Event] = None


@dataclass
class SearchResult:
    move: Optional[Move]
    score: float
    stats: SearchStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_score(score: float) -> str:
    """Human readable evaluation: ``+245`` or ``Mate in 2``."""
    if abs(score) > MATE_THRESHOLD:
        plies = CHECKMATE_SCORE - abs(score)
        moves = int((plies + 1) // 2)
        return f"Mate in {moves}" if score > 0 else f"Mated in {moves}"
    return f"{int(round(score)):+d}"


def _check_cancel(ctx: SearchContext) -> None:
    if ctx.cancel_event is not None and not (ctx.stats.nodes & 255) and ctx.cancel_event.is_set():
        raise SearchCancelled()


def _is_rule_draw(board: Board) -> bool:
    """Fifty-move rule or a repeated position inside the search tree."""
    return board.halfmove_clock >= 100 or board.repetition_count() >= 2


def _terminal_value(board: Board, in_check: bool, ctx: SearchContext, ply: int) -> float:
    """Value of a position with no legal moves."""
    if in_check:
        mate = CHECKMATE_SCORE - ply
        return -mate if board.turn == ctx.ai_color else mate
    return 0.0


def _to_tt(score: float, ply: int) -> float:
    """Make a mate score independent of the ply at which it was found."""
    if score > MATE_THRESHOLD:
        return score + ply
    if score < -MATE_THRESHOLD:
        return score - ply
    return score


def _from_tt(score: float, ply: int) -> float:
    if score > MATE_THRESHOLD:
        return score - ply
    if score < -MATE_THRESHOLD:
        return score + ply
    return score


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------
def order_moves(board: Board, moves: List[Move], tt_move_key=None, detect_checks: bool = True) -> List[Move]:
    """Sort ``moves`` so the most promising are searched first.

    Order: checkmating moves, transposition-table move, captures (MVV-LVA),
    promotions, checking moves, then everything else.  Good ordering makes
    alpha-beta cut off earlier, so far fewer positions are searched.

    MVV-LVA = *Most Valuable Victim, Least Valuable Attacker*: capturing a
    queen with a pawn is tried before capturing a pawn with a queen.
    """
    scored = []
    for move in moves:
        score = 0
        if tt_move_key is not None and move.key() == tt_move_key:
            score += TT_BONUS
        if move.captured_piece is not None:
            score += CAPTURE_BONUS + _RANK[move.captured_piece[1]] * 100 - _RANK[move.piece[1]]
        if move.promotion:
            score += PROMOTION_BONUS + PIECE_VALUES[move.promotion]
        if detect_checks:
            make_move(board, move)
            if is_in_check(board, board.turn):           # the opponent is now in check
                score += CHECK_BONUS
                if not generate_legal_moves(board, board.turn):
                    score += MATE_BONUS                     # ...and has no reply: checkmate
            undo_move(board, move)
        scored.append((score, move))
    scored.sort(key=lambda pair: -pair[0])                # stable: ties keep generation order
    return [m for _, m in scored]


# ---------------------------------------------------------------------------
# Plain minimax (reference implementation, no pruning)
# ---------------------------------------------------------------------------
def minimax(board: Board, depth: int, maximizing_player: bool,
            ctx: Optional[SearchContext] = None, ply: int = 0) -> float:
    """Full-width minimax value of ``board``.

    ``maximizing_player`` is True when it is the AI's turn to move.  Every
    legal move is explored -- there is no pruning -- so this is mainly useful
    as a correctness reference for :func:`alpha_beta` and to show how many more
    nodes an unpruned search needs.
    """
    if ctx is None:
        ctx = SearchContext()
    stats = ctx.stats
    stats.nodes += 1
    _check_cancel(ctx)

    if ply > 0 and _is_rule_draw(board):
        return 0.0
    in_check = is_in_check(board, board.turn)
    if depth <= 0:
        if in_check and ply < ctx.max_ply:
            depth = 1                                        # check extension
        else:
            return evaluate_board(board, ctx.ai_color)

    moves = generate_legal_moves(board, board.turn)
    if not moves:
        return _terminal_value(board, in_check, ctx, ply)

    if maximizing_player:
        best = -INFINITY
        for move in moves:
            make_move(board, move)
            best = max(best, minimax(board, depth - 1, False, ctx, ply + 1))
            undo_move(board, move)
    else:
        best = INFINITY
        for move in moves:
            make_move(board, move)
            best = min(best, minimax(board, depth - 1, True, ctx, ply + 1))
            undo_move(board, move)
    return best


# ---------------------------------------------------------------------------
# Minimax with alpha-beta pruning
# ---------------------------------------------------------------------------
def alpha_beta(board: Board, depth: int, alpha: float, beta: float, maximizing_player: bool,
               ctx: Optional[SearchContext] = None, ply: int = 0) -> float:
    """Minimax with alpha-beta pruning (fail-soft) and a transposition table.

    ``alpha`` = the best score the *maximiser* is already guaranteed on the
    path to this node; ``beta`` = the best score the *minimiser* is already
    guaranteed.  If ``alpha >= beta`` the minimiser would never let the game
    reach this node (it has a better alternative elsewhere), so the remaining
    moves cannot affect the final answer and are skipped.
    """
    if ctx is None:
        ctx = SearchContext()
    stats = ctx.stats
    stats.nodes += 1
    _check_cancel(ctx)

    if ply > 0 and _is_rule_draw(board):
        return 0.0
    in_check = is_in_check(board, board.turn)
    if depth <= 0:
        if in_check and ply < ctx.max_ply:
            depth = 1                                        # check extension
        else:
            return evaluate_board(board, ctx.ai_color)

    alpha_original = alpha
    tt_move_key = None
    tt = ctx.tt
    if tt is not None:
        entry = tt.probe(board.hash)
        if entry is not None:
            stats.tt_hits += 1
            tt_depth, tt_score, tt_flag, tt_move_key = entry
            if tt_depth >= depth:
                tt_score = _from_tt(tt_score, ply)
                if tt_flag == EXACT:
                    return tt_score
                if tt_flag == LOWER:
                    alpha = max(alpha, tt_score)
                else:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

    moves = generate_legal_moves(board, board.turn)
    if not moves:
        return _terminal_value(board, in_check, ctx, ply)
    if ctx.use_ordering:
        # Checking every move for check/mate is only worth it when enough
        # depth remains below to profit from the better ordering.
        moves = order_moves(board, moves, tt_move_key, detect_checks=depth >= 2)

    best_move_key = None
    count = len(moves)
    if maximizing_player:
        value = -INFINITY
        for i, move in enumerate(moves):
            make_move(board, move)
            score = alpha_beta(board, depth - 1, alpha, beta, False, ctx, ply + 1)
            undo_move(board, move)
            if score > value:
                value = score
                best_move_key = move.key()
            if value > alpha:
                alpha = value
            if alpha >= beta:                # <-- BETA CUTOFF: minimiser avoids this node
                stats.cutoffs += 1
                stats.pruned += count - i - 1
                break
    else:
        value = INFINITY
        for i, move in enumerate(moves):
            make_move(board, move)
            score = alpha_beta(board, depth - 1, alpha, beta, True, ctx, ply + 1)
            undo_move(board, move)
            if score < value:
                value = score
                best_move_key = move.key()
            if value < beta:
                beta = value
            if alpha >= beta:                # <-- ALPHA CUTOFF: maximiser avoids this node
                stats.cutoffs += 1
                stats.pruned += count - i - 1
                break

    if tt is not None:
        if value <= alpha_original:
            flag = UPPER                      # fail-low: real value is <= value
        elif value >= beta:
            flag = LOWER                      # fail-high: real value is >= value
        else:
            flag = EXACT
        tt.store(board.hash, depth, _to_tt(value, ply), flag, best_move_key)
    return value


# ---------------------------------------------------------------------------
# Root driver
# ---------------------------------------------------------------------------
def find_best_move(
    board: Board,
    depth: int = SEARCH_DEPTH,
    ai_color: Optional[str] = None,
    use_alpha_beta: bool = True,
    use_transposition: bool = True,
    use_ordering: bool = True,
    randomize: bool = False,
    cancel_event: Optional[threading.Event] = None,
    debug: Optional[bool] = None,
) -> SearchResult:
    """Search ``board`` to ``depth`` plies and return the best move.

    The board is mutated during the search but restored before returning.
    ``randomize`` shuffles equally-ranked moves first so the AI does not play
    the identical game every time.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    ai_color = ai_color or board.turn
    debug = DEBUG if debug is None else debug
    started = time.perf_counter()

    ctx = SearchContext(
        ai_color=ai_color,
        tt=TranspositionTable() if (use_transposition and use_alpha_beta) else None,
        use_ordering=use_ordering,
        max_ply=depth + CHECK_EXTENSION_PLIES,
        cancel_event=cancel_event,
    )
    stats = ctx.stats
    stats.depth = depth

    moves = generate_legal_moves(board, board.turn)
    stats.root_moves = len(moves)
    if not moves:
        stats.elapsed = time.perf_counter() - started
        return SearchResult(None, 0.0, stats)

    if randomize:
        random.shuffle(moves)
    if use_ordering:
        moves = order_moves(board, moves, detect_checks=True)

    root_maximizing = board.turn == ai_color
    alpha, beta = -INFINITY, INFINITY
    best_move: Optional[Move] = None
    best_score = -INFINITY if root_maximizing else INFINITY

    for move in moves:
        make_move(board, move)
        try:
            if use_alpha_beta:
                score = alpha_beta(board, depth - 1, alpha, beta, not root_maximizing, ctx, 1)
            else:
                score = minimax(board, depth - 1, not root_maximizing, ctx, 1)
        finally:
            undo_move(board, move)
        if root_maximizing:
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)

    stats.best_score = best_score
    stats.elapsed = time.perf_counter() - started
    if ctx.tt is not None:
        stats.tt_hits = ctx.tt.hits
    result = SearchResult(best_move, best_score, stats)
    if debug:
        print_debug(result)
    return result


def print_debug(result: SearchResult) -> None:
    """Print the statistics block described in the project brief."""
    s = result.stats
    print(f"AI depth: {s.depth}")
    print(f"Generated moves: {s.root_moves}")
    print(f"Nodes searched: {s.nodes}")
    print(f"Nodes pruned: {s.pruned}")
    print(f"Best evaluation: {format_score(result.score)}")
    if result.move is not None:
        print(f"AI move: {result.move.describe()}")
    print(f"Time taken: {s.elapsed:.2f} seconds")
