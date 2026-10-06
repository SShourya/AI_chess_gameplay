"""Headless tests of the Pygame front-end (SDL "dummy" video driver).

They drive the real ``ChessApp`` with synthetic mouse/keyboard events, so they
cover input handling, the AI thread hand-off, promotion, restart and resizing.
"""

import os
import random
import time
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from ai.minimax import find_best_move  # noqa: E402
from game.game_state import GameStatus  # noqa: E402
from main import ChessApp, MIN_WINDOW_SIZE  # noqa: E402
from ui.board_renderer import BoardRenderer, PieceRenderer, compute_layout  # noqa: E402


def sq(name):
    return 8 - int(name[1]), "abcdefgh".index(name[0])


class UITestCase(unittest.TestCase):
    """Base class for the headless UI tests.

    pygame.init()/pygame.quit() are surprisingly fragile when cycled dozens
    of times in one process under SDL's "dummy" video/font drivers (this is a
    quirk of repeatedly tearing down and reinitialising SDL subsystems in a
    single process, not something a normal single-run application ever hits),
    so the whole test module shares one pygame session and each test just
    gets a fresh ChessApp + window.
    """

    def setUp(self):
        self.app = ChessApp("Easy")            # depth 2 keeps tests quick

    def tearDown(self):
        self.app.worker.cancel()
        deadline = time.time() + 2.0
        while self.app.worker.busy and time.time() < deadline:
            time.sleep(0.01)


    # -- helpers --------------------------------------------------------------
    def layout(self):
        return compute_layout(*pygame.display.get_surface().get_size())

    def click(self, pos):
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))

    def click_square(self, name):
        self.click(BoardRenderer.square_rect(self.layout(), sq(name)).center)

    def key(self, key):
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))

    def pump_until(self, predicate, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            self.app.update()
            self.app.draw()
            if predicate():
                return True
            time.sleep(0.01)
        return False


def setUpModule():
    pygame.init()


def tearDownModule():
    pygame.quit()


class InteractionTests(UITestCase):
    def test_click_select_then_move_and_ai_replies(self):
        self.click_square("e2")
        self.assertEqual(self.app.interaction.selected, sq("e2"))
        self.assertEqual(set(self.app.interaction.targets), {sq("e3"), sq("e4")})
        self.click_square("e4")
        self.assertEqual(self.app.state.move_log, ["e2e4"])
        self.assertTrue(self.pump_until(lambda: self.app.state.turn == "w"), "AI never replied")
        self.assertEqual(len(self.app.state.move_log), 2)
        self.assertIsNotNone(self.app.ai_info.stats)
        self.assertGreater(self.app.ai_info.stats.nodes, 0)

    def test_illegal_click_destination_is_ignored(self):
        self.click_square("e2")
        self.click_square("e5")                      # not a legal target: just changes/clears selection
        self.assertEqual(self.app.state.move_log, [])
        self.assertEqual(self.app.state.turn, "w")

    def test_cannot_select_opponent_pieces(self):
        self.click_square("e7")
        self.assertIsNone(self.app.interaction.selected)

    def test_clicking_another_own_piece_switches_selection(self):
        self.click_square("e2")
        self.click_square("g1")
        self.assertEqual(self.app.interaction.selected, sq("g1"))
        self.assertEqual(set(self.app.interaction.targets), {sq("f3"), sq("h3")})

    def test_escape_clears_selection(self):
        self.click_square("e2")
        self.key(pygame.K_ESCAPE)
        self.assertIsNone(self.app.interaction.selected)

    def test_human_is_locked_out_while_ai_thinks(self):
        self.click_square("e2")
        self.click_square("e4")
        self.app.update()                            # AI search starts
        self.assertTrue(self.app.worker.busy)
        self.click_square("d2")
        self.assertIsNone(self.app.interaction.selected)
        self.assertEqual(len(self.app.state.move_log), 1)

    def test_promotion_popup_and_choice(self):
        self.app.state.load_fen("7k/P7/8/8/8/8/8/K7 w - - 0 1")
        self.click_square("a7")
        self.click_square("a8")
        self.assertEqual(self.app.interaction.pending_promotion, (sq("a7"), sq("a8")))
        self.assertEqual(self.app.state.move_log, [])            # nothing played yet
        rect = dict(BoardRenderer.promotion_choices(self.layout(), "w"))["r"]
        self.click(rect.center)
        self.assertEqual(self.app.state.board.grid[0][0], "wr")
        self.assertEqual(self.app.state.move_log, ["a7a8r"])

    def test_promotion_can_be_cancelled(self):
        self.app.state.load_fen("7k/P7/8/8/8/8/8/K7 w - - 0 1")
        self.click_square("a7")
        self.click_square("a8")
        self.key(pygame.K_ESCAPE)
        self.assertIsNone(self.app.interaction.pending_promotion)
        self.assertEqual(self.app.state.board.grid[1][0], "wp")

    def test_difficulty_selector_changes_depth(self):
        seg = self.app.panel.difficulty
        self.app.draw()                              # lays the panel out
        for name, depth in (("Hard", 4), ("Easy", 2), ("Medium", 3)):
            target = [r for n, r in seg._segments() if n == name][0]
            self.click(target.center)
            self.assertEqual(self.app.difficulty, name)
            self.assertEqual(self.app.search_depth, depth)


class LifecycleTests(UITestCase):
    def test_restart_button_resets_everything(self):
        self.click_square("e2")
        self.click_square("e4")
        self.pump_until(lambda: len(self.app.state.move_log) == 2)
        self.app.draw()
        self.click(self.app.panel.restart_button.rect.center)
        self.assertEqual(self.app.state.move_log, [])
        self.assertEqual(self.app.state.captured, {"w": [], "b": []})
        self.assertIsNone(self.app.ai_info.stats)
        self.assertEqual(self.app.state.turn, "w")

    def test_restart_during_ai_search_discards_the_stale_move(self):
        self.app.depth_override = 4
        self.click_square("e2")
        self.click_square("e4")
        self.app.update()
        self.assertTrue(self.app.worker.busy)
        self.key(pygame.K_r)
        self.assertFalse(self.app.worker.busy)
        time.sleep(0.8)                              # give the old thread time to finish
        for _ in range(5):
            self.app.update()
        self.assertEqual(self.app.state.move_log, [])            # nothing from the old game leaked in
        self.assertEqual(self.app.state.board.to_fen().split()[0],
                         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    def test_game_over_dialog_play_again(self):
        for u in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            self.assertTrue(self.app.state.play(sq(u[:2]), sq(u[2:4])))
        self.assertEqual(self.app.state.status, GameStatus.CHECKMATE)
        self.app.draw()
        self.click_square("e2")                      # board is locked when the game is over
        self.assertIsNone(self.app.interaction.selected)
        self.click(self.app.dialog.play_again.rect.center)
        self.assertFalse(self.app.state.game_over)
        self.assertEqual(self.app.state.move_log, [])

    def test_quit_stops_the_loop(self):
        self.app.draw()
        self.click(self.app.panel.quit_button.rect.center)
        self.assertFalse(self.app.running)

    def test_ai_does_not_move_after_game_over(self):
        self.app.state.load_fen("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
        for _ in range(5):
            self.app.update()
        self.assertFalse(self.app.worker.busy)
        self.assertEqual(self.app.state.move_log, [])


class LayoutTests(UITestCase):
    def test_layout_stays_square_and_inside_window_for_common_sizes(self):
        for w, h in ((900, 640), (1080, 740), (1280, 720), (1366, 768), (1920, 1080), (1000, 1400)):
            with self.subTest(size=(w, h)):
                lay = compute_layout(w, h)
                self.assertEqual(lay.board_rect.width, lay.board_rect.height)
                self.assertEqual(lay.board_rect.width, lay.square * 8)
                window = pygame.Rect(0, 0, w, h)
                self.assertTrue(window.contains(lay.board_rect))
                self.assertTrue(window.contains(lay.panel_rect))
                self.assertFalse(lay.board_rect.colliderect(lay.panel_rect))

    def test_resize_event_is_clamped_to_minimum_and_redraws(self):
        self.app.handle_event(pygame.event.Event(pygame.VIDEORESIZE, size=(500, 400), w=500, h=400))
        w, h = pygame.display.get_surface().get_size()
        self.assertGreaterEqual(w, MIN_WINDOW_SIZE[0])
        self.assertGreaterEqual(h, MIN_WINDOW_SIZE[1])
        self.app.draw()
        pygame.display.set_mode((1400, 900), pygame.RESIZABLE)
        self.app.draw()
        self.click_square("e2")                      # hit-testing follows the new layout
        self.assertEqual(self.app.interaction.selected, sq("e2"))

    def test_both_piece_render_modes_produce_surfaces_for_all_pieces(self):
        for force_vector in (True, False):
            renderer = PieceRenderer(force_vector)
            for colour in "wb":
                for t in "pnbrqk":
                    surf = renderer.get(colour + t, 64)
                    self.assertEqual(surf.get_size(), (64, 64))
                    self.assertIsNotNone(surf.get_bounding_rect())

    def test_every_status_screen_draws_without_error(self):
        fens = {
            "playing": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "check": "4k3/8/8/8/8/8/8/4RK2 b - - 0 1",
            "mate": "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",
            "stalemate": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
            "draw": "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
        }
        for name, fen in fens.items():
            with self.subTest(status=name):
                self.app.state.load_fen(fen)
                self.app.draw()


class FullGameTests(UITestCase):
    def test_random_human_versus_ai_game_never_breaks_rules(self):
        """A 'human' clicking random legal moves against the real AI thread."""
        rng = random.Random(7)
        start = time.time()
        plies = 0
        while not self.app.state.game_over and plies < 60 and time.time() - start < 90:
            gs = self.app.state
            if gs.is_human_turn and not self.app.worker.busy and self.app.pending_result is None:
                move = rng.choice(gs.legal_moves)
                self.click_square(move_name(move.from_square))
                self.click_square(move_name(move.to_square))
                if self.app.interaction.pending_promotion:
                    rect = dict(BoardRenderer.promotion_choices(self.layout(), "w"))["q"]
                    self.click(rect.center)
                plies += 1
            self.app.update()
            self.app.draw()
            time.sleep(0.002)
            self.assertEqual(gs.board.hash, gs.board.compute_hash())
        self.assertGreater(len(self.app.state.move_log), 10)
        # replay the log on a fresh board: proves every move (human and AI) was legal
        from game.game_state import GameState
        replay = GameState()
        for uci in self.app.state.move_log:
            promo = uci[4] if len(uci) == 5 else None
            self.assertTrue(replay.play(sq(uci[:2]), sq(uci[2:4]), promo), f"illegal move in log: {uci}")
        self.assertEqual(replay.board.to_fen(), self.app.state.board.to_fen())


def move_name(square):
    return "abcdefgh"[square[1]] + str(8 - square[0])


if __name__ == "__main__":
    unittest.main()
