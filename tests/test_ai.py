"""Tests for the search, evaluation and AI plumbing."""

import threading
import time
import unittest

from ai.evaluation import CHECKMATE_SCORE, evaluate_board, terminal_score
from ai.minimax import (
    DIFFICULTY_DEPTHS, SearchCancelled, alpha_beta, find_best_move, format_score,
    minimax, order_moves, SearchContext,
)
from ai.transposition import EXACT, TranspositionTable
from ai.worker import AIWorker
from game.board import Board
from game.game_state import GameState, GameStatus
from game.move import make_move, undo_move
from game.rules import generate_legal_moves, is_checkmate, is_stalemate
from tests.helpers import board

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
POSITIONS = [
    START,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
]


def mirror_fen(fen: str) -> str:
    """Colour-flip a position (ranks reversed, colours swapped, side swapped)."""
    parts = fen.split()
    rows = [r.swapcase() for r in reversed(parts[0].split("/"))]
    turn = "b" if parts[1] == "w" else "w"
    castling = "".join(sorted(parts[2].swapcase(), key=lambda ch: "KQkq".index(ch))) if parts[2] != "-" else "-"
    ep = "-" if parts[3] == "-" else parts[3][0] + str(9 - int(parts[3][1]))
    return f"{'/'.join(rows)} {turn} {castling} {ep} {parts[4]} {parts[5]}"


class EvaluationTests(unittest.TestCase):
    def test_start_position_is_balanced(self):
        self.assertEqual(evaluate_board(board(START), "w"), 0)

    def test_material_advantage_is_positive_for_the_side_ahead(self):
        up_a_queen = board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        self.assertGreater(evaluate_board(up_a_queen, "w"), 800)
        self.assertLess(evaluate_board(up_a_queen, "b"), -800)

    def test_perspective_flips_the_sign(self):
        b = board(POSITIONS[1])
        self.assertEqual(evaluate_board(b, "w"), -evaluate_board(b, "b"))

    def test_evaluation_is_colour_symmetric(self):
        for fen in POSITIONS:
            with self.subTest(fen=fen):
                a = evaluate_board(board(fen), "w")
                m = evaluate_board(board(mirror_fen(fen)), "w")
                self.assertEqual(a, -m)

    def test_piece_square_tables_prefer_the_centre(self):
        centre = board("4k3/p7/8/8/3N4/8/P7/4K3 w - - 0 1")
        rim = board("4k3/p7/8/8/8/8/P7/N3K3 w - - 0 1")
        self.assertGreater(evaluate_board(centre, "w"), evaluate_board(rim, "w"))

    def test_doubled_and_isolated_pawns_are_penalised(self):
        healthy = board("4k3/8/8/8/8/3PP3/8/4K3 w - - 0 1")
        doubled = board("4k3/8/8/8/8/4P3/4P3/4K3 w - - 0 1")
        self.assertGreater(evaluate_board(healthy, "w"), evaluate_board(doubled, "w"))

    def test_passed_pawn_bonus(self):
        passed = board("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1")
        blocked = board("4k3/4p3/8/4P3/8/8/8/4K3 w - - 0 1")
        self.assertGreater(evaluate_board(passed, "w"), evaluate_board(blocked, "w") + 100)

    def test_castled_king_scores_better_than_uncastled(self):
        base = "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ%s w - - 0 1"
        castled = board(base % "1RK1")
        uncastled = board(base % "K2R")
        self.assertGreater(evaluate_board(castled, "w"), evaluate_board(uncastled, "w"))

    def test_insufficient_material_evaluates_to_zero(self):
        self.assertEqual(evaluate_board(board("8/8/8/4k3/8/8/8/3BK3 w - - 0 1")), 0)

    def test_terminal_scores(self):
        mated = board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
        self.assertEqual(terminal_score(mated, "w", 0), CHECKMATE_SCORE)      # AI (white) delivered mate
        self.assertEqual(terminal_score(mated, "b", 0), -CHECKMATE_SCORE)     # AI (black) got mated
        stale = board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(terminal_score(stale, "b", 0), 0)
        self.assertIsNone(terminal_score(board(START), "w", 0))
        self.assertEqual(evaluate_board(mated, "w", detect_terminal=True), CHECKMATE_SCORE)

    def test_faster_mate_scores_higher(self):
        mated = board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
        self.assertGreater(terminal_score(mated, "w", 1), terminal_score(mated, "w", 3))
        self.assertLess(terminal_score(mated, "b", 1), terminal_score(mated, "b", 3))

    def test_format_score(self):
        self.assertEqual(format_score(245), "+245")
        self.assertEqual(format_score(-30), "-30")
        self.assertEqual(format_score(CHECKMATE_SCORE - 1), "Mate in 1")
        self.assertEqual(format_score(-(CHECKMATE_SCORE - 4)), "Mated in 2")


class SearchEquivalenceTests(unittest.TestCase):
    def test_alpha_beta_returns_same_value_as_minimax(self):
        for fen in POSITIONS:
            b = board(fen)
            for depth in (1, 2):
                for ai in ("w", "b"):
                    with self.subTest(fen=fen[:20], depth=depth, ai=ai):
                        mm = find_best_move(b, depth, ai, use_alpha_beta=False)
                        ab = find_best_move(b, depth, ai, use_transposition=False)
                        tt = find_best_move(b, depth, ai)
                        ab_unordered = find_best_move(b, depth, ai, use_ordering=False, use_transposition=False)
                        self.assertEqual(mm.score, ab.score)
                        self.assertEqual(mm.score, tt.score)
                        self.assertEqual(mm.score, ab_unordered.score)

    def test_alpha_beta_move_is_optimal_according_to_minimax(self):
        b = board(POSITIONS[2])
        depth, ai = 2, "b"
        ab = find_best_move(b, depth, ai)
        ctx = SearchContext(ai_color=ai, max_ply=depth + 4)
        values = {}
        for m in generate_legal_moves(b, b.turn):
            make_move(b, m)
            values[m.uci()] = minimax(b, depth - 1, False, ctx, 1)
            undo_move(b, m)
        self.assertEqual(values[ab.move.uci()], max(values.values()))

    def test_pruning_actually_reduces_the_search(self):
        b = board(POSITIONS[1])
        full = find_best_move(b, 2, "w", use_alpha_beta=False)
        pruned = find_best_move(b, 2, "w", use_transposition=False)
        self.assertLess(pruned.stats.nodes, full.stats.nodes / 3)
        self.assertGreater(pruned.stats.pruned, 0)
        self.assertGreater(pruned.stats.cutoffs, 0)

    def test_move_ordering_reduces_nodes(self):
        b = board(POSITIONS[1])
        ordered = find_best_move(b, 3, "w", use_transposition=False)
        unordered = find_best_move(b, 3, "w", use_transposition=False, use_ordering=False)
        self.assertEqual(ordered.score, unordered.score)
        self.assertLess(ordered.stats.nodes, unordered.stats.nodes)

    def test_transposition_table_does_not_change_the_value_at_depth_3(self):
        b = board(POSITIONS[2])
        with_tt = find_best_move(b, 3, "b")
        without = find_best_move(b, 3, "b", use_transposition=False)
        self.assertEqual(with_tt.score, without.score)

    def test_search_restores_the_board(self):
        for fen in POSITIONS:
            b = board(fen)
            h, f, hist = b.hash, b.to_fen(), list(b.history)
            find_best_move(b, 3)
            self.assertEqual((b.hash, b.to_fen(), b.history), (h, f, hist))

    def test_module_level_alpha_beta_matches_minimax_signature_usage(self):
        b = board(POSITIONS[3])
        ctx1 = SearchContext(ai_color="w", max_ply=6)
        ctx2 = SearchContext(ai_color="w", max_ply=6)
        v1 = minimax(b, 2, True, ctx1)
        v2 = alpha_beta(b, 2, float("-inf"), float("inf"), True, ctx2)
        self.assertEqual(v1, v2)


class TacticalTests(unittest.TestCase):
    def test_ai_takes_a_free_queen(self):
        b = board("4k3/8/8/3q4/3Q4/8/8/4K3 b - - 0 1")
        result = find_best_move(b, 2, "b")
        self.assertEqual(result.move.uci(), "d5d4")

    def test_ai_prefers_capturing_the_bigger_piece(self):
        # White rook attacks a defended pawn and a hanging queen
        b = board("4k3/8/8/1q1p4/8/8/8/1R2K3 w - - 0 1")
        self.assertEqual(find_best_move(b, 2, "w").move.uci(), "b1b5")

    def test_ai_finds_mate_in_one_as_white(self):
        b = board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        r = find_best_move(b, 2, "w")
        self.assertEqual(r.move.uci(), "a1a8")
        self.assertEqual(r.score, CHECKMATE_SCORE - 1)

    def test_ai_finds_mate_in_one_as_black(self):
        b = board("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")
        r = find_best_move(b, 2, "b")
        self.assertEqual(r.move.uci(), "a8a1")
        self.assertEqual(r.score, CHECKMATE_SCORE - 1)

    def test_ai_finds_mate_in_two(self):
        # Ladder mate: Ra7+ / Kg8 / Rb8#  (white rooks a1, b2? -> classic two-rook ladder)
        b = board("7k/8/8/8/8/8/1R6/R5K1 w - - 0 1")
        r = find_best_move(b, 3, "w")
        self.assertGreater(r.score, CHECKMATE_SCORE - 10)
        make_move(b, r.move)
        reply = find_best_move(b, 2, "b")
        make_move(b, reply.move)
        final = find_best_move(b, 1, "w")
        make_move(b, final.move)
        self.assertTrue(is_checkmate(b, "b"))

    def test_ai_prefers_the_faster_mate(self):
        b = board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        r = find_best_move(b, 4, "w")
        self.assertEqual(r.move.uci(), "a1a8")          # mate now beats any slower mate

    def test_ai_avoids_walking_into_mate_in_one(self):
        # Black to move; White threatens Ra8#.  Whatever black plays, white must have no mating reply.
        b = board("6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1")
        r = find_best_move(b, 3, "b")
        make_move(b, r.move)
        for reply in generate_legal_moves(b, "w"):
            make_move(b, reply)
            self.assertFalse(is_checkmate(b, "b"), f"AI allowed mate after {r.move} / {reply}")
            undo_move(b, reply)

    def test_ai_sees_that_a_defence_exists(self):
        b = board("6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1")
        r = find_best_move(b, 3, "b")
        self.assertGreater(r.score, -CHECKMATE_SCORE / 2)        # not scored as lost

    def test_root_minimises_when_the_side_to_move_is_the_opponent(self):
        # White (the "opponent" here) has mate in one; from Black-AI's view that is the worst outcome.
        b = board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
        r = find_best_move(b, 1, "b")
        self.assertEqual(r.move.uci(), "a1a8")
        self.assertEqual(r.score, -(CHECKMATE_SCORE - 1))
        self.assertEqual(format_score(r.score), "Mated in 1")

    def test_ai_does_not_stalemate_when_winning(self):
        # K+Q v K.  Qg6 would be stalemate; Qg7 is mate.  The AI must find the mate.
        b = board("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1")
        r = find_best_move(b, 3, "w")
        make_move(b, r.move)
        self.assertFalse(is_stalemate(b, "b"))
        self.assertTrue(is_checkmate(b, "b"))


class LegalityGuaranteeTests(unittest.TestCase):
    def test_ai_only_ever_plays_legal_moves_in_self_play(self):
        for ai_depths in ((2, 2), (1, 2)):
            gs = GameState()
            for ply in range(40):
                if gs.game_over:
                    break
                color = gs.turn
                depth = ai_depths[0] if color == "w" else ai_depths[1]
                result = find_best_move(gs.board.copy(), depth, color, randomize=True)
                self.assertIsNotNone(result.move)
                legal = {m.key() for m in gs.legal_moves}
                self.assertIn(result.move.key(), legal)
                self.assertTrue(gs.play(result.move.from_square, result.move.to_square, result.move.promotion))
            self.assertEqual(gs.board.hash, gs.board.compute_hash())

    def test_game_state_rejects_illegal_moves(self):
        gs = GameState()
        self.assertFalse(gs.play((6, 4), (3, 4)))         # e2-e5 is not legal
        self.assertFalse(gs.play((7, 4), (6, 4)))         # king onto own pawn
        self.assertFalse(gs.play((3, 3), (4, 3)))         # empty square
        self.assertEqual(gs.board.to_fen(), START)

    def test_no_move_when_game_is_over(self):
        b = board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
        result = find_best_move(b, 2, "b")
        self.assertIsNone(result.move)


class InfrastructureTests(unittest.TestCase):
    def test_difficulty_depths(self):
        self.assertEqual(DIFFICULTY_DEPTHS, {"Easy": 2, "Medium": 3, "Hard": 4})

    def test_transposition_table_basics(self):
        tt = TranspositionTable(max_entries=3)
        tt.store(1, 3, 10.0, EXACT, None)
        self.assertEqual(tt.probe(1), (3, 10.0, EXACT, None))
        tt.store(1, 2, 99.0, EXACT, None)                 # shallower result must not overwrite
        self.assertEqual(tt.probe(1)[1], 10.0)
        self.assertIsNone(tt.probe(42))
        for k in (2, 3, 4):
            tt.store(k, 1, 0.0, EXACT, None)
        self.assertLessEqual(len(tt), 3)

    def test_order_moves_puts_mate_then_captures_first(self):
        b = board("6k1/5ppp/8/3p4/8/8/8/R2Q2K1 w - - 0 1")
        ordered = order_moves(b, generate_legal_moves(b, "w"))
        self.assertEqual(ordered[0].uci(), "a1a8")        # 1. checkmate
        self.assertEqual(ordered[1].uci(), "d1d5")        # 2. the only capture
        self.assertTrue(all(not m.is_capture for m in ordered[2:]))

    def test_mvv_lva_prefers_cheap_attacker_on_valuable_victim(self):
        b = board("4k3/8/8/3q4/2P5/8/8/3RK3 w - - 0 1")
        ordered = order_moves(b, generate_legal_moves(b, "w"), detect_checks=False)
        self.assertEqual(ordered[0].uci(), "c4d5")        # pawn takes queen before rook takes queen (d1xd5)
        self.assertEqual(ordered[1].uci(), "d1d5")

    def test_cancelled_search_raises(self):
        ev = threading.Event()
        ev.set()
        with self.assertRaises(SearchCancelled):
            find_best_move(board(START), 4, "w", cancel_event=ev)

    def test_worker_returns_a_legal_move_without_blocking(self):
        w = AIWorker()
        b = board(START)
        w.start(b, 2, "w")
        self.assertTrue(w.busy)
        result = None
        deadline = time.time() + 20
        while result is None and time.time() < deadline:
            result = w.poll()
            time.sleep(0.01)
        self.assertIsNotNone(result)
        self.assertIn(result.move.key(), {m.key() for m in generate_legal_moves(b, "w")})
        self.assertFalse(w.busy)
        self.assertEqual(b.to_fen(), START)               # the UI's board was never touched

    def test_worker_cancel_discards_result(self):
        w = AIWorker()
        w.start(board(START), 4, "w")
        w.cancel()
        self.assertFalse(w.busy)
        time.sleep(0.2)
        self.assertIsNone(w.poll())

    def test_game_state_status_text(self):
        gs = GameState()
        gs.load_fen("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
        self.assertEqual(gs.status, GameStatus.CHECKMATE)
        self.assertIn("White wins", gs.status_text())


if __name__ == "__main__":
    unittest.main()
