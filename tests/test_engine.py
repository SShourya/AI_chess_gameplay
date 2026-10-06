"""Whole-engine integrity checks: perft counts, exact undo, Zobrist hashing."""

import os
import random
import unittest

from game.board import Board, START_FEN
from game.move import make_move, undo_move
from game.rules import generate_legal_moves, is_in_check, perft
from tests.helpers import board

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION_3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION_4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION_5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"

SLOW = os.environ.get("CHESS_SLOW_TESTS") == "1"


class PerftTests(unittest.TestCase):
    """Published node counts; a single wrong count means a move-generation bug."""

    def check(self, fen, expected):
        b = board(fen)
        for depth, count in enumerate(expected, start=1):
            self.assertEqual(perft(b, depth), count, f"depth {depth} of {fen}")

    def test_start_position(self):
        self.check(START_FEN, [20, 400, 8902, 197281])

    def test_kiwipete_castling_ep_promotion_heavy(self):
        self.check(KIWIPETE, [48, 2039, 97862])

    def test_position_3_en_passant_pins_and_checks(self):
        self.check(POSITION_3, [14, 191, 2812, 43238])

    def test_position_4_promotions_and_castling_rights(self):
        self.check(POSITION_4, [6, 264, 9467])

    def test_position_5(self):
        self.check(POSITION_5, [44, 1486, 62379])

    @unittest.skipUnless(SLOW, "set CHESS_SLOW_TESTS=1 to run")
    def test_deeper_counts(self):
        self.check(START_FEN, [20, 400, 8902, 197281, 4865609])
        self.check(KIWIPETE, [48, 2039, 97862, 4085603])


class FenTests(unittest.TestCase):
    def test_round_trip(self):
        for fen in (START_FEN, KIWIPETE, POSITION_3, POSITION_4, POSITION_5,
                    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"):
            self.assertEqual(Board.from_fen(fen).to_fen(), fen)

    def test_initial_board_matches_the_documented_layout(self):
        b = Board.initial()
        self.assertEqual(b.grid[0], ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"])
        self.assertEqual(b.grid[1], ["bp"] * 8)
        self.assertEqual(b.grid[6], ["wp"] * 8)
        self.assertEqual(b.grid[7], ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"])
        self.assertEqual(b.grid[3], [None] * 8)

    def test_bad_fen_raises(self):
        for bad in ("", "8/8/8 w - -", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w KQkq - 0 1"):
            with self.assertRaises(ValueError):
                Board.from_fen(bad)


class MakeUndoTests(unittest.TestCase):
    def snapshot(self, b):
        return (b.to_fen(), b.hash, [row[:] for row in b.grid], dict(b.king_pos), list(b.history))

    def test_undo_restores_every_kind_of_move_exactly(self):
        for fen in (START_FEN, KIWIPETE, POSITION_3, POSITION_4, POSITION_5):
            b = board(fen)
            before = self.snapshot(b)
            for move in generate_legal_moves(b, b.turn):
                make_move(b, move)
                undo_move(b, move)
                self.assertEqual(self.snapshot(b), before, f"{move} in {fen}")

    def test_random_playouts_keep_hash_and_state_consistent(self):
        rng = random.Random(1234)
        for fen in (START_FEN, KIWIPETE, POSITION_4):
            for _ in range(6):
                b = board(fen)
                played = []
                for _ply in range(60):
                    moves = generate_legal_moves(b, b.turn)
                    if not moves:
                        break
                    m = rng.choice(moves)
                    make_move(b, m)
                    played.append(m)
                    self.assertEqual(b.hash, b.compute_hash(), "incremental hash drifted")
                    self.assertFalse(is_in_check(b, "b" if b.turn == "w" else "w"),
                                     "mover left own king in check")
                    self.assertEqual(Board.from_fen(b.to_fen()).hash, b.hash)
                start = board(fen)
                for m in reversed(played):
                    undo_move(b, m)
                self.assertEqual(b.to_fen(), start.to_fen())
                self.assertEqual(b.hash, start.hash)
                self.assertEqual(len(b.history), 1)

    def test_legal_move_generation_leaves_board_untouched(self):
        b = board(KIWIPETE)
        before = self.snapshot(b)
        generate_legal_moves(b, "w")
        generate_legal_moves(b, "b")
        self.assertEqual(self.snapshot(b), before)

    def test_transposed_move_orders_give_the_same_hash(self):
        def play(seq):
            b = Board.initial()
            for u in seq:
                m = next(m for m in generate_legal_moves(b, b.turn) if m.uci() == u)
                make_move(b, m)
            return b
        a = play(["g1f3", "g8f6", "b1c3", "b8c6"])
        c = play(["b1c3", "b8c6", "g1f3", "g8f6"])
        self.assertEqual(a.hash, c.hash)
        self.assertNotEqual(a.hash, Board.initial().hash)

    def test_copy_is_independent(self):
        b = board(START_FEN)
        c = b.copy()
        make_move(c, generate_legal_moves(c, "w")[0])
        self.assertEqual(b.to_fen(), START_FEN)
        self.assertNotEqual(b.hash, c.hash)


if __name__ == "__main__":
    unittest.main()
