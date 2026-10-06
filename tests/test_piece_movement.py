"""Movement rules for every piece type."""

import unittest

from game.board import Board
from game.move import make_move, undo_move
from game.rules import generate_legal_moves
from tests.helpers import board, legal_uci, targets

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class PawnTests(unittest.TestCase):
    def test_white_single_and_double_step_from_start(self):
        self.assertEqual(targets(board(START), "e2"), {"e3", "e4"})

    def test_black_pawn_moves_down_the_board(self):
        b = board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        self.assertEqual(targets(b, "d7"), {"d6", "d5"})

    def test_pawn_off_start_rank_moves_one_square_only(self):
        self.assertEqual(targets(board("7k/8/8/8/8/4P3/8/K7 w - - 0 1"), "e3"), {"e4"})

    def test_double_step_blocked_by_piece_on_far_square(self):
        b = board("7k/8/8/8/4n3/8/4P3/K7 w - - 0 1")
        self.assertEqual(targets(b, "e2"), {"e3"})

    def test_pawn_fully_blocked(self):
        self.assertEqual(targets(board("7k/8/8/8/8/4n3/4P3/K7 w - - 0 1"), "e2"), set())

    def test_pawn_captures_diagonally(self):
        b = board("7k/8/8/3p1p2/4P3/8/8/K7 w - - 0 1")
        self.assertEqual(targets(b, "e4"), {"e5", "d5", "f5"})

    def test_pawn_cannot_capture_straight_ahead(self):
        b = board("7k/8/8/3ppp2/4P3/8/8/K7 w - - 0 1")
        self.assertEqual(targets(b, "e4"), {"d5", "f5"})

    def test_pawn_cannot_capture_own_piece(self):
        b = board("7k/8/8/3PPP2/4P3/8/8/K7 w - - 0 1")
        self.assertEqual(targets(b, "e4"), set())

    def test_white_promotion_offers_four_pieces(self):
        b = board("7k/P7/8/8/8/8/8/K7 w - - 0 1")
        promos = {u for u in legal_uci(b) if u.startswith("a7")}
        self.assertEqual(promos, {"a7a8q", "a7a8r", "a7a8b", "a7a8n"})

    def test_promotion_by_capture(self):
        b = board("1r5k/P7/8/8/8/8/8/K7 w - - 0 1")
        promos = {u for u in legal_uci(b) if u.startswith("a7")}
        self.assertEqual(len(promos), 8)
        self.assertIn("a7b8q", promos)
        self.assertIn("a7a8n", promos)

    def test_black_promotion(self):
        b = board("k7/8/8/8/8/8/7p/K7 b - - 0 1")
        promos = {u for u in legal_uci(b) if u.startswith("h2")}
        self.assertEqual(promos, {"h2h1q", "h2h1r", "h2h1b", "h2h1n"})

    def test_promotion_replaces_pawn_and_undo_restores_it(self):
        b = board("7k/P7/8/8/8/8/8/K7 w - - 0 1")
        move = next(m for m in generate_legal_moves(b, "w") if m.uci() == "a7a8q")
        before = b.to_fen()
        make_move(b, move)
        self.assertEqual(b.grid[0][0], "wq")
        self.assertIsNone(b.grid[1][0])
        undo_move(b, move)
        self.assertEqual(b.grid[1][0], "wp")
        self.assertEqual(b.to_fen(), before)


class KnightTests(unittest.TestCase):
    def test_centre_knight_has_eight_moves(self):
        b = board("k7/8/8/8/3N4/8/8/7K w - - 0 1")
        self.assertEqual(targets(b, "d4"), {"b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"})

    def test_knight_jumps_over_pieces_from_start(self):
        b = board(START)
        self.assertEqual(targets(b, "b1"), {"a3", "c3"})
        self.assertEqual(targets(b, "g1"), {"f3", "h3"})

    def test_corner_knight_has_two_moves(self):
        self.assertEqual(targets(board("k7/8/8/8/8/8/8/N6K w - - 0 1"), "a1"), {"b3", "c2"})

    def test_knight_captures_enemy_but_not_friend(self):
        b = board("k7/8/2p5/8/3N4/8/8/5P1K w - - 0 1")
        t = targets(b, "d4")
        self.assertIn("c6", t)
        b2 = board("k7/8/2P5/8/3N4/8/8/7K w - - 0 1")
        self.assertNotIn("c6", targets(b2, "d4"))


class BishopTests(unittest.TestCase):
    def test_bishop_on_empty_board_has_thirteen_moves(self):
        b = board("k7/8/8/8/3B4/8/8/7K w - - 0 1")
        self.assertEqual(len(targets(b, "d4")), 13)

    def test_bishop_only_moves_diagonally(self):
        t = targets(board("k7/8/8/8/3B4/8/8/7K w - - 0 1"), "d4")
        self.assertTrue(all(abs(ord(s[0]) - ord("d")) == abs(int(s[1]) - 4) for s in t))

    def test_bishop_blocked_by_own_piece_and_stops_at_capture(self):
        b = board("k7/8/5p2/8/3B4/2P5/8/7K w - - 0 1")
        t = targets(b, "d4")
        self.assertIn("f6", t)                       # capture allowed
        self.assertNotIn("g7", t)                    # ...but cannot jump past it
        self.assertNotIn("c3", t)                    # own pawn
        self.assertNotIn("b2", t)                    # beyond own pawn


class RookTests(unittest.TestCase):
    def test_rook_horizontal_and_vertical_fourteen_moves(self):
        b = board("k7/8/8/8/4R3/8/8/7K w - - 0 1")
        t = targets(b, "e4")
        self.assertEqual(len(t), 14)
        self.assertTrue({"a4", "h4", "e1", "e8"} <= t)

    def test_rook_blocked_by_own_pawn(self):
        b = board("k7/8/4P3/8/4R3/8/8/7K w - - 0 1")
        t = targets(b, "e4")
        self.assertIn("e5", t)
        self.assertNotIn("e6", t)
        self.assertNotIn("e7", t)


class QueenTests(unittest.TestCase):
    def test_queen_combines_rook_and_bishop(self):
        b = board("k7/8/8/8/3Q4/8/8/7K w - - 0 1")
        t = targets(b, "d4")
        self.assertEqual(len(t), 27)
        self.assertTrue({"d8", "a4", "h8", "a1", "g1", "a7"} <= t)   # file, rank, both diagonals

    def test_queen_cannot_jump(self):
        t = targets(board(START), "d1")
        self.assertEqual(t, set())


class KingTests(unittest.TestCase):
    def test_king_moves_one_square_in_every_direction(self):
        b = board("k7/8/8/8/4K3/8/8/8 w - - 0 1")
        self.assertEqual(targets(b, "e4"), {"d3", "e3", "f3", "d4", "f4", "d5", "e5", "f5"})

    def test_king_cannot_move_into_check(self):
        b = board("3r3k/8/8/8/8/8/8/4K3 w - - 0 1")
        self.assertEqual(targets(b, "e1"), {"e2", "f1", "f2"})   # d1/d2 are on the rook's file

    def test_kings_cannot_stand_next_to_each_other(self):
        b = board("8/8/8/4k3/8/4K3/8/8 w - - 0 1")
        self.assertEqual(targets(b, "e3"), {"d2", "e2", "f2", "d3", "f3"})

    def test_king_can_capture_undefended_piece_but_not_defended_one(self):
        self.assertIn("e2", targets(board("k7/8/8/8/8/8/4r3/4K3 w - - 0 1"), "e1"))
        self.assertNotIn("e2", targets(board("k3r3/8/8/8/8/8/4r3/4K3 w - - 0 1"), "e1"))


if __name__ == "__main__":
    unittest.main()
