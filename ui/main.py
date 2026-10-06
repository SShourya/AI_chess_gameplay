"""Chess AI - play White against a Minimax / Alpha-Beta computer opponent.

Run with::

    python main.py                       # medium difficulty
    python main.py --difficulty Hard     # Easy | Medium | Hard
    python main.py --debug               # print search statistics to the console
    python main.py --depth 5             # override the search depth

Keys: R / N = restart, Q = quit, Esc = cancel selection.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import pygame

import ai.minimax as minimax_module
from ai.minimax import DIFFICULTY_DEPTHS, SEARCH_DEPTH, SearchResult
from ai.worker import AIWorker
from game.game_state import GameState
from ui.board_renderer import BACKGROUND, BoardRenderer, PieceRenderer, compute_layout
from ui.input_handler import InputHandler, Interaction
from ui.menu import AIInfo, GameOverDialog, SidePanel

WINDOW_SIZE = (1080, 740)
MIN_WINDOW_SIZE = (900, 640)
FPS = 60
MIN_THINK_SECONDS = 0.35        # so instant replies don't feel jarring


class ChessApp:
    """Owns the window, the game state and the AI, and runs the main loop."""

    def __init__(self, difficulty: str = "Medium", depth_override: Optional[int] = None,
                 force_vector_pieces: Optional[bool] = None) -> None:
        pygame.init()
        pygame.display.set_caption("Chess AI - Minimax & Alpha-Beta")
        self.screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.difficulty = difficulty
        self.depth_override = depth_override
        self.running = True

        self.state = GameState()
        self.worker = AIWorker()
        self.ai_info = AIInfo()
        self.pending_result: Optional[SearchResult] = None
        self.interaction = Interaction()

        pieces = PieceRenderer(force_vector_pieces)
        self.renderer = BoardRenderer(pieces)
        self.panel = SidePanel(pieces, difficulty, self.restart, self.quit, self.set_difficulty)
        self.dialog = GameOverDialog(self.restart, self.quit)
        self.input = InputHandler(
            self.state, self.renderer, self.panel, self.dialog, self.interaction,
            is_ai_busy=lambda: self.worker.busy or self.pending_result is not None,
            on_restart=self.restart, on_quit=self.quit,
        )

    # ------------------------------------------------------------------ actions
    def restart(self) -> None:
        """Abort any search and start a fresh game."""
        self.worker.cancel()
        self.pending_result = None
        self.ai_info = AIInfo()
        self.interaction.clear()
        self.state.new_game()

    def quit(self) -> None:
        self.worker.cancel()
        self.running = False

    def set_difficulty(self, name: str) -> None:
        self.difficulty = name              # takes effect on the computer's next move

    @property
    def search_depth(self) -> int:
        return self.depth_override or DIFFICULTY_DEPTHS.get(self.difficulty, SEARCH_DEPTH)

    # ------------------------------------------------------------------- update
    def update(self) -> None:
        """Advance the computer player (never blocks the UI thread)."""
        gs = self.state
        if not gs.is_ai_turn:
            self.ai_info.thinking = False
            return

        if not self.worker.busy and self.pending_result is None:
            self.ai_info.thinking = True
            self.ai_info.thinking_since = time.time()
            self.worker.start(gs.board, self.search_depth, gs.ai_color)
            return

        if self.pending_result is None:
            self.pending_result = self.worker.poll()
        if self.pending_result is not None and time.time() - self.ai_info.thinking_since >= MIN_THINK_SECONDS:
            result, self.pending_result = self.pending_result, None
            self._apply_ai_result(result)

    def _apply_ai_result(self, result: SearchResult) -> None:
        gs = self.state
        move = result.move
        # Every move -- even the AI's -- must pass GameState.play's legality gate.
        if move is None or not gs.play(move.from_square, move.to_square, move.promotion):
            print("WARNING: AI returned no/illegal move; playing first legal move", file=sys.stderr)
            fallback = gs.legal_moves[0]
            gs.play(fallback.from_square, fallback.to_square, fallback.promotion)
        self.ai_info.thinking = False
        self.ai_info.stats = result.stats
        self.ai_info.score = result.score
        self.ai_info.move_text = move.describe() if move else ""
        self.interaction.clear()

    # --------------------------------------------------------------------- draw
    def draw(self) -> None:
        surface = pygame.display.get_surface()
        layout = compute_layout(*surface.get_size())
        mouse = pygame.mouse.get_pos()
        surface.fill(BACKGROUND)
        gs, ui = self.state, self.interaction
        self.renderer.draw(surface, layout, gs, ui.selected, ui.targets, mouse)
        self.panel.draw(surface, layout.panel_rect, gs, self.ai_info, mouse)
        if ui.pending_promotion is not None:
            self.renderer.draw_promotion_popup(surface, layout, gs.human_color, mouse)
        if gs.game_over:
            self.dialog.draw(surface, layout, gs, mouse)
        pygame.display.flip()

    # -------------------------------------------------------------------- events
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.VIDEORESIZE:
            w, h = max(event.w, MIN_WINDOW_SIZE[0]), max(event.h, MIN_WINDOW_SIZE[1])
            if (w, h) != (event.w, event.h):
                pygame.display.set_mode((w, h), pygame.RESIZABLE)
            return
        layout = compute_layout(*pygame.display.get_surface().get_size())
        self.input.handle_event(event, layout)

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play chess against a Minimax/Alpha-Beta AI.")
    parser.add_argument("--difficulty", choices=list(DIFFICULTY_DEPTHS), default="Medium")
    parser.add_argument("--depth", type=int, help="override the search depth (plies)")
    parser.add_argument("--debug", action="store_true", help="print search statistics to the console")
    parser.add_argument("--unicode-pieces", action="store_true",
                        help="try Unicode chess glyphs instead of the default hand-drawn pieces "
                             "(falls back automatically if no font has real chess glyphs)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    minimax_module.DEBUG = args.debug
    ChessApp(args.difficulty, args.depth, False if args.unicode_pieces else None).run()


if __name__ == "__main__":
    main()
