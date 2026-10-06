"""Special rules: castling, en passant, check handling, mate, stalemate, draws."""

import unittest

from game.game_state import GameState, GameStatus
from game.move import make_move
from game.rules import (
    generate_legal_moves, generate_pseudo_legal_moves, is_checkmate,
    is_in_check, is_insufficient_material, is_square_attacked, is_stalemate,
)
from tests.helpers import board, legal_uci, targets


def play(gs: GameState, *ucis: str) -> None:
    """Play moves given as UCI text through the GameState gate."""
    for u in ucis:
        promo = u[4] if len(u) == 5 else None
        frm = (8 - int(u[1]), "abcdefgh".index(u[0]))
        to = (8 - int(u[3]), "abcdefgh".index(u[2]))
        assert gs.play(frm, to, promo), f"illegal move in test: {u}"


class CastlingTests(unittest.TestCase):
    BOTH = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

    def test_white_can_castle_both_sides(self):
        moves = legal_uci(board(self.BOTH))
        self.assertIn("e1g1", moves)
        self.assertIn("e1c1", moves)

    def test_black_can_castle_both_sides(self):
        moves = legal_uci(board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"))
        self.assertIn("e8g8", moves)
        self.assertIn("e8c8", moves)

    def test_kingside_castling_moves_king_and_rook(self):
        b = board(self.BOTH)
        m = next(m for m in generate_legal_moves(b, "w") if m.uci() == "e1g1")
        make_move(b, m)
        self.assertEqual((b.grid[7][6], b.grid[7][5]), ("wk", "wr"))
        self.assertIsNone(b.grid[7][7])
        self.assertIsNone(b.grid[7][4])

    def test_queenside_castling_moves_king_and_rook(self):
        b = board(self.BOTH)
        m = next(m for m in generate_legal_moves(b, "w") if m.uci() == "e1c1")
        make_move(b, m)
        self.assertEqual((b.grid[7][2], b.grid[7][3]), ("wk", "wr"))
        self.assertIsNone(b.grid[7][0])

    def test_no_castling_without_rights(self):
        moves = legal_uci(board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"))
        self.assertNotIn("e1g1", moves)
        self.assertNotIn("e1c1", moves)

    def test_only_the_side_with_rights_can_castle(self):
        moves = legal_uci(board("r3k2r/8/8/8/8/8/8/R3K2R w Kkq - 0 1"))
        self.assertIn("e1g1", moves)
        self.assertNotIn("e1c1", moves)

    def test_no_castling_with_pieces_between(self):
        moves = legal_uci(board("r3k2r/8/8/8/8/8/8/RN2K1NR w KQkq - 0 1"))
        self.assertNotIn("e1g1", moves)
        self.assertNotIn("e1c1", moves)

    def test_queenside_needs_b_file_empty_too(self):
        moves = legal_uci(board("r3k2r/8/8/8/8/8/8/RN2K2R w KQkq - 0 1"))
        self.assertNotIn("e1c1", moves)
        self.assertIn("e1g1", moves)

    def test_no_castling_out_of_check(self):
        moves = legal_uci(board("4r1k1/8/8/8/8/8/8/R3K2R w KQ - 0 1"))
        self.assertNotIn("e1g1", moves)
        self.assertNotIn("e1c1", moves)

    def test_no_castling_through_attacked_square(self):
        # black rook on f8 attacks f1 -> kingside illegal, queenside fine
        moves = legal_uci(board("5rk1/8/8/8/8/8/8/R3K2R w KQ - 0 1"))
        self.assertNotIn("e1g1", moves)
        self.assertIn("e1c1", moves)

    def test_no_castling_into_check(self):
        # black rook on g8 attacks g1
        moves = legal_uci(board("6rk/8/8/8/8/8/8/R3K2R w KQ - 0 1"))
        self.assertNotIn("e1g1", moves)
        self.assertIn("e1c1", moves)

    def test_queenside_allowed_when_only_b_file_is_attacked(self):
        # b1 is attacked but the king never crosses it
        moves = legal_uci(board("1r5k/8/8/8/8/8/8/R3K2R w KQ - 0 1"))
        self.assertIn("e1c1", moves)

    def test_rights_lost_after_king_or_rook_moves(self):
        gs = GameState()
        gs.load_fen(self.BOTH)
        play(gs, "h1h2", "a8a7")
        self.assertEqual(gs.board.castling & 1, 0)       # white kingside gone
        self.assertEqual(gs.board.castling & 8, 0)       # black queenside gone
        play(gs, "e1e2", "e8e7")
        self.assertEqual(gs.board.castling, 0)

    def test_rights_lost_when_rook_is_captured(self):
        gs = GameState()
        gs.load_fen("r3k2r/8/8/8/8/8/6b1/R3K2R b KQkq - 0 1")
        play(gs, "g2h1")                                  # bishop takes the h1 rook
        self.assertEqual(gs.board.castling & 1, 0)

    def test_castling_via_game_state_and_undo_hash_consistency(self):
        gs = GameState()
        gs.load_fen(self.BOTH)
        play(gs, "e1g1")
        self.assertEqual(gs.board.hash, gs.board.compute_hash())


class EnPassantTests(unittest.TestCase):
    def test_valid_en_passant_available_after_double_push(self):
        gs = GameState()
        gs.load_fen("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
        play(gs, "d7d5")
        self.assertIn("e5d6", {m.uci() for m in gs.legal_moves})

    def test_en_passant_removes_the_captured_pawn(self):
        gs = GameState()
        gs.load_fen("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
        play(gs, "d7d5", "e5d6")
        self.assertIsNone(gs.board.grid[3][3])            # d5 pawn is gone
        self.assertEqual(gs.board.grid[2][3], "wp")       # capturer stands on d6
        self.assertEqual(gs.captured["w"], ["bp"])

    def test_en_passant_only_immediately_after_the_double_push(self):
        gs = GameState()
        gs.load_fen("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
        play(gs, "d7d5", "e1e2", "e8e7")                  # both sides make another move
        self.assertNotIn("e5d6", {m.uci() for m in gs.legal_moves})

    def test_no_en_passant_after_single_step_pawn_moves(self):
        gs = GameState()
        gs.load_fen("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1")
        play(gs, "d7d6")
        self.assertIsNone(gs.board.en_passant)
        # e5xd6 is still legal, but as an ordinary capture, not en passant
        move = gs.find_legal_move((3, 4), (2, 3))
        self.assertIsNotNone(move)
        self.assertFalse(move.is_en_passant)

    def test_black_can_capture_en_passant_too(self):
        gs = GameState()
        gs.load_fen("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1")
        play(gs, "e2e4")
        self.assertIn("d4e3", {m.uci() for m in gs.legal_moves})
        play(gs, "d4e3")
        self.assertIsNone(gs.board.grid[4][4])            # e4 pawn removed

    def test_en_passant_illegal_when_it_exposes_own_king(self):
        # After exd6 e.p. both pawns leave rank 5 and the rook checks the king.
        b = board("8/8/8/K2pP2r/8/8/8/7k w - d6 0 1")
        moves = legal_uci(b)
        self.assertNotIn("e5d6", moves)
        self.assertIn("e5e6", moves)

    def test_en_passant_target_recorded_only_if_capturable(self):
        gs = GameState()
        gs.load_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        play(gs, "e2e4")
        self.assertIsNone(gs.board.en_passant)


class CheckTests(unittest.TestCase):
    def test_detect_check_from_each_piece_type(self):
        cases = {
            "rook": "4r2k/8/8/8/8/8/8/4K3 w - - 0 1",
            "bishop": "7k/8/8/8/7b/8/8/4K3 w - - 0 1",
            "queen": "7k/8/8/8/8/8/8/q3K3 w - - 0 1",
            "knight": "7k/8/8/8/8/3n4/8/4K3 w - - 0 1",
            "pawn": "7k/8/8/8/8/8/3p4/4K3 w - - 0 1",
        }
        for name, fen in cases.items():
            with self.subTest(piece=name):
                self.assertTrue(is_in_check(board(fen), "w"))

    def test_no_check_when_blocked(self):
        b = board("4r2k/8/8/8/8/4P3/8/4K3 w - - 0 1")
        self.assertFalse(is_in_check(b, "w"))

    def test_is_square_attacked(self):
        b = board("7k/8/8/8/3R4/8/8/K7 w - - 0 1")
        self.assertTrue(is_square_attacked(b, (4, 0), "w"))     # a4 on the rook's rank
        self.assertTrue(is_square_attacked(b, (0, 3), "w"))     # d8 on the rook's file
        self.assertFalse(is_square_attacked(b, (3, 4), "w"))    # e5 is diagonal

    def test_must_resolve_check(self):
        # White king e1 checked by rook e8; legal replies: move king, or block with Re2/Be2.
        b = board("4r2k/8/8/8/8/8/3B4/R3K3 w - - 0 1")
        moves = legal_uci(b)
        self.assertEqual({m for m in moves if not m.startswith("e1")}, {"d2e3"})
        self.assertTrue({"e1d1", "e1f1", "e1f2"} <= moves)
        self.assertNotIn("a1a2", moves)

    def test_capture_the_checking_piece(self):
        b = board("7k/8/8/8/8/8/4r3/R3K3 w - - 0 1")
        # rook on e2 gives check; Kxe2 legal since it is undefended
        self.assertIn("e1e2", legal_uci(b))

    def test_pinned_piece_cannot_leave_the_pin_line(self):
        b = board("4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1")
        self.assertEqual(targets(b, "e2"), set())                # knight pinned, no moves at all

    def test_pinned_rook_may_move_along_the_pin(self):
        b = board("4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1")
        self.assertEqual(targets(b, "e2"), {"e3", "e4", "e5", "e6", "e7", "e8"})

    def test_pseudo_legal_includes_moves_that_legal_filter_removes(self):
        b = board("4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1")
        pseudo = {m.uci() for m in generate_pseudo_legal_moves(b, "w") if m.from_square == (6, 4)}
        self.assertTrue(pseudo)                                  # knight has pseudo-legal moves
        self.assertEqual(targets(b, "e2"), set())                # but none are legal

    def test_double_check_forces_king_move(self):
        b = board("4r2k/8/8/8/8/5n2/8/4K3 w - - 0 1")
        for m in generate_legal_moves(b, "w"):
            self.assertEqual(m.piece, "wk")


class MateAndStalemateTests(unittest.TestCase):
    def test_back_rank_mate(self):
        b = board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
        self.assertTrue(is_checkmate(b, "b"))

    def test_fools_mate(self):
        gs = GameState()
        play(gs, "f2f3", "e7e5", "g2g4", "d8h4")
        self.assertEqual(gs.status, GameStatus.CHECKMATE)
        self.assertEqual(gs.winner, "b")
        self.assertTrue(gs.game_over)

    def test_scholars_mate(self):
        gs = GameState()
        play(gs, "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7")
        self.assertEqual(gs.status, GameStatus.CHECKMATE)
        self.assertEqual(gs.winner, "w")

    def test_smothered_mate(self):
        gs = GameState()
        gs.load_fen("6rk/5Npp/8/8/8/8/8/K7 b - - 0 1")     # Nf7# : the king is boxed in by its own rook and pawns
        self.assertEqual(gs.status, GameStatus.CHECKMATE)

    def test_check_that_is_not_mate_is_reported_as_check(self):
        gs = GameState()
        gs.load_fen("4k3/8/8/8/8/8/8/4RK2 b - - 0 1")
        self.assertEqual(gs.status, GameStatus.CHECK)
        self.assertFalse(gs.game_over)

    def test_known_stalemate(self):
        b = board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(is_stalemate(b, "b"))
        self.assertFalse(is_in_check(b, "b"))
        gs = GameState()
        gs.load_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(gs.status, GameStatus.STALEMATE)
        self.assertIsNone(gs.winner)

    def test_stalemate_is_not_checkmate(self):
        b = board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertFalse(is_checkmate(b, "b"))


class DrawTests(unittest.TestCase):
    def test_insufficient_material_cases(self):
        yes = ["8/8/8/4k3/8/8/8/4K3 w - - 0 1", "8/8/8/4k3/8/8/8/3BK3 w - - 0 1",
               "8/8/8/4k3/8/8/8/3NK3 w - - 0 1", "8/8/8/3bk3/8/8/8/3BK3 w - - 0 1"]
        no = ["8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", "8/8/8/4k3/8/8/8/3RK3 w - - 0 1",
              "8/8/8/4k1b1/8/8/8/3BK3 w - - 0 1", "8/8/8/4k3/8/8/8/2NNK3 w - - 0 1"]
        for fen in yes:
            with self.subTest(fen=fen):
                self.assertTrue(is_insufficient_material(board(fen)))
        for fen in no:
            with self.subTest(fen=fen):
                self.assertFalse(is_insufficient_material(board(fen)))

    def test_fifty_move_rule(self):
        gs = GameState()
        gs.load_fen("4k3/8/8/8/8/8/R7/4K3 w - - 99 60")
        play(gs, "a2a3")
        self.assertEqual(gs.status, GameStatus.DRAW)
        self.assertIn("Fifty", gs.draw_reason)

    def test_threefold_repetition(self):
        gs = GameState()
        gs.load_fen("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
        for _ in range(2):
            play(gs, "a2a3", "e8d8", "a3a2", "d8e8")
        self.assertEqual(gs.status, GameStatus.DRAW)
        self.assertIn("repetition", gs.draw_reason.lower())


if __name__ == "__main__":
    unittest.main()
